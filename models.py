import torch
import torch.nn as nn
import torch.nn.functional as F
from selfonn import SelfONN1d as SelfONN1dlayer


#___________________________________________________________________________________________________________________________
class DownsampleConvBlock(nn.Module):
    """Convolutional block with optional downsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=False, dropout=False, q=3):
        super().__init__()
        self.dropout = dropout
        self.conv = SelfONN1dlayer(in_channels, out_channels, kernel_size, stride, padding, q=q)
        self.relu = nn.Tanh()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.downsample = downsample
        if downsample:
            self.downsample_layer = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_layer = nn.Dropout(0.25)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout_layer(out)

        if self.downsample:
            out = self.downsample_layer(out)

        return out

#___________________________________________________________________________________________________________________________

class UpsampleConvBlock(nn.Module):
    """Block with upsampling followed by a convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, upsample=False, dropout=False, q=3):
        super().__init__()
        self.dropout = dropout
        self.conv = SelfONN1dlayer(in_channels, out_channels, kernel_size, stride, padding, q=q)
        self.relu = nn.Tanh()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout_layer = nn.Dropout(0.25)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout_layer(out)

        return out
    

#___________________________________________________________________________________________________________________________
# Generator and Discriminator with simple downsample and upsample blocks.
#___________________________________________________________________________________________________________________________
class Generator(nn.Module):
    def __init__(self,q):
        super().__init__()

        # Downsampling
        self.time_down1 = DownsampleConvBlock(2, 16, downsample=True, q=q)
        self.time_down2 = DownsampleConvBlock(16, 32, downsample=True, q=q)
        self.time_down3 = DownsampleConvBlock(32, 64, downsample=True, dropout=True, q=q)
        self.time_down4 = DownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        self.time_down5 = DownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        
        # Upsampling 
        self.time_up1 = UpsampleConvBlock(64, 64, upsample=True, dropout=True, q=q)
        self.time_up2 = UpsampleConvBlock(64, 64, upsample=True, dropout=True, q=q)
        self.time_up3 = UpsampleConvBlock(64, 32, upsample=True, q=q)
        self.time_up4 = UpsampleConvBlock(32, 16, upsample=True, q=q)
        self.time_up5 = UpsampleConvBlock(16, 16, upsample=True, q=q)

        # Output Branch
        self.time_output_conv = nn.Conv1d(16, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, time_input):
        # Encoder
        time_features_d1 = self.time_down1(time_input) # 16 x 512
        time_features_d2 = self.time_down2(time_features_d1) # 32 x 256       
        time_features_d3 = self.time_down3(time_features_d2) # 64 x 128
        time_features_d4 = self.time_down4(time_features_d3) # 64 x 64
        time_features_d5 = self.time_down5(time_features_d4) # 64 x 32 # Bottleneck

        # Decoder
        time_features_u1 = self.time_up1(time_features_d5) # 64 x 64
        time_features_u2 = self.time_up2(time_features_u1) # 64 x 128
        time_features_u3 = self.time_up3(time_features_u2) # 32 x 256
        time_features_u4 = self.time_up4(time_features_u3) # 16 x 512
        time_features_u5 = self.time_up5(time_features_u4) # 16 x 1024

        # output
        time_output = self.time_output_conv(time_features_u5) # 2 x 1024

        return time_output
    

class Discriminator(nn.Module):
    def __init__(self,q):
        super().__init__()
        # Time domain branch with downsampling
        self.time_down1 = DownsampleConvBlock(2, 16, downsample=True, q=q)
        self.time_down2 = DownsampleConvBlock(16, 32, downsample=True, q=q)
        self.time_down3 = DownsampleConvBlock(32, 32, downsample=True, q=q)
        self.time_down4 = DownsampleConvBlock(32, 64, downsample=True, dropout=True, q=q)
        self.time_down5 = DownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        self.time_down6 = DownsampleConvBlock(64, 16, downsample=True, q=q)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            #nn.Sigmoid()
        )
        
    def forward(self, time_input):
        # Process time-domain input
        time_features = self.time_down1(time_input) # 16 x 512
        time_features = self.time_down2(time_features) # 32 x 256
        time_features = self.time_down3(time_features) # 32 x 128
        time_features = self.time_down4(time_features) # 64 x 64
        time_features = self.time_down5(time_features) # 64 x 32
        time_features = self.time_down6(time_features) # 64 x 16
        
        # Classify combined features as real or fake
        classification = self.classifier(time_features)

        return classification
    
###########################################################################################################
###########################################################################################################

class ResidualDownsampleConvBlock(nn.Module):
    """Residual convolutional block with optional downsampling, returns output before activation for skip connections."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=False, dropout=False, q=3):
        super().__init__()
        self.dropout = dropout
        self.conv = SelfONN1dlayer(in_channels, out_channels, kernel_size, stride, padding,q=q)
        self.relu = nn.Tanh()
        #self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.downsample = downsample
        if downsample:
            self.downsample_layer = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout_layer = nn.Dropout(0.25)

        # Adapt the input to have the same shape as the output if necessary
        self.adapt_identity = SelfONN1dlayer(in_channels, out_channels, kernel_size=1, stride=stride,q=1) if in_channels != out_channels or stride != 1 else None
        

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.norm(out)

        # Adapt the identity (skip connection)
        if self.adapt_identity:
            identity = self.adapt_identity(identity)

        # Combine with the skip connection
        out += identity

        # Save the output before activation
        output_before_activation = out

        # Apply the activation function
        out = self.relu(out)

        # Apply dropout after activation
        if self.dropout:
            out = self.dropout_layer(out)

        if self.downsample:
            out = self.downsample_layer(out)
            output_before_activation = self.downsample_layer(output_before_activation)

        return out, output_before_activation

#___________________________________________________________________________________________________________________________    

class ResidualUpsampleConvBlock(nn.Module):
    """Residual block with upsampling followed by a convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, upsample=False, dropout=False, q=3):
        super().__init__()
        self.dropout = dropout
        self.conv = SelfONN1dlayer(in_channels, out_channels, kernel_size, stride, padding,q=q)
        self.relu = nn.Tanh()
        #self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout_layer = nn.Dropout(0.25)

        # Adapt the input to have the same number of channels as the output if necessary
        self.adapt_identity = SelfONN1dlayer(in_channels, out_channels, kernel_size=1, q=1) if in_channels != out_channels else None
        SelfONN1dlayer(in_channels, out_channels, kernel_size=1, stride=stride,q=1)

    def forward(self, x):
        
        # Upsample the main path if required
        if self.upsample:
            x = self.upsample_layer(x)

        identity = x

        # Adapt the number of channels of the identity if necessary
        if self.adapt_identity:
            identity = self.adapt_identity(identity)

        x = self.conv(x)
        x = self.norm(x)

        # Add the skip connection
        x += identity
        x = self.relu(x)

        # Apply dropout after activation
        if self.dropout:
            x = self.dropout_layer(x)

        return x
    

#___________________________________________________________________________________________________________________________
# Generator and Discriminator with residual downsample and upsample blocks.
#___________________________________________________________________________________________________________________________
class ResidualGenerator(nn.Module):
    def __init__(self,q):
        super().__init__()

        # Downsampling
        self.time_down1 = ResidualDownsampleConvBlock(2, 16, downsample=True, q=q)
        self.time_down2 = ResidualDownsampleConvBlock(16, 32, downsample=True, q=q)
        self.time_down3 = ResidualDownsampleConvBlock(32, 64, downsample=True, dropout=True, q=q)
        self.time_down4 = ResidualDownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        self.time_down5 = ResidualDownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        
        # Upsampling 
        self.time_up1 = ResidualUpsampleConvBlock(64, 64, upsample = True, dropout=True, q=q)
        self.time_up2 = ResidualUpsampleConvBlock(64+64, 64, upsample = True, dropout=True, q=q)
        self.time_up3 = ResidualUpsampleConvBlock(64+64, 32, upsample = True, q=q)
        self.time_up4 = ResidualUpsampleConvBlock(32+32, 16, upsample=True, q=q)
        self.time_up5 = ResidualUpsampleConvBlock(16+16, 16, upsample=True, q=q)

        # Output Branch
        self.time_output_conv = nn.Conv1d(16, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, time_input):
        # Encoder
        time_features_d1, time_features_d1_skip = self.time_down1(time_input) # 16 x 512
        time_features_d2, time_features_d2_skip = self.time_down2(time_features_d1) # 32 x 256       
        time_features_d3, time_features_d3_skip = self.time_down3(time_features_d2) # 64 x 128
        time_features_d4, time_features_d4_skip = self.time_down4(time_features_d3) # 64 x 64

        time_features_d5, latent_vector = self.time_down5(time_features_d4) # 64 x 32 # Bottleneck

        # Decoder
        merged_features_u1 = self.time_up1(time_features_d5) # 64 x 64
        merged_features_u1_c = torch.cat([merged_features_u1, time_features_d4_skip], dim=1) # 64+64 x 64
        merged_features_u2 = self.time_up2(merged_features_u1_c) # 64 x 128
        merged_features_u2_c = torch.cat([merged_features_u2, time_features_d3_skip], dim=1) # 64+64 x 128
        merged_features_u3 = self.time_up3(merged_features_u2_c) # 32 x 256
        merged_features_u3_c = torch.cat([merged_features_u3, time_features_d2_skip], dim=1) # 32+32 x 256
        merged_features_u4 = self.time_up4(merged_features_u3_c) # 16 x 512
        merged_features_u4_c = torch.cat([merged_features_u4, time_features_d1_skip], dim=1) # 16+16 x 512
        merged_features_u5 = self.time_up5(merged_features_u4_c) # 16 x 1024
            
        # output
        time_output = self.time_output_conv(merged_features_u5) # 2 x 1024

        return time_output
    
#___________________________________________________________________________________________________________________________
class ResidualDiscriminator(nn.Module):
    def __init__(self,q):
        super().__init__()
        # Time domain branch with downsampling
        self.time_down1 = ResidualDownsampleConvBlock(2, 16, downsample=True, q=q)
        self.time_down2 = ResidualDownsampleConvBlock(16, 32, downsample=True, q=q)
        self.time_down3 = ResidualDownsampleConvBlock(32, 32, downsample=True, q=q)
        self.time_down4 = ResidualDownsampleConvBlock(32, 64, downsample=True, dropout=True, q=q)
        self.time_down5 = ResidualDownsampleConvBlock(64, 64, downsample=True, dropout=True, q=q)
        self.time_down6 = ResidualDownsampleConvBlock(64, 16, downsample=True, q=q)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            #nn.Sigmoid()
        )
        
    def forward(self, time_input):
        # Process time-domain input
        time_features, _ = self.time_down1(time_input) # 16 x 512
        time_features, _  = self.time_down2(time_features) # 32 x 256
        time_features, _  = self.time_down3(time_features) # 32 x 128
                
        time_features, _  = self.time_down4(time_features) # 64 x 64
        time_features, _  = self.time_down5(time_features) # 64 x 32
        time_features, _  = self.time_down6(time_features) # 64 x 16
        
        # Classify combined features as real or fake
        classification = self.classifier(time_features)

        return classification
##########################################################################################################
##########################################################################################################


if __name__ == '__main__':
    # Quick shape and parameter-count check
    x = torch.randn(4, 2, 1024)
    for q in (1, 3):
        G, D = ResidualGenerator(q=q), ResidualDiscriminator(q=q)
        print(f"q={q}: G out {tuple(G(x).shape)}, D out {tuple(D(x).shape)}, "
              f"G params {sum(p.numel() for p in G.parameters())/1e3:.2f}K, "
              f"D params {sum(p.numel() for p in D.parameters())/1e3:.2f}K")

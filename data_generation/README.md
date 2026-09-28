# Radar Signal Dataset Generation (MATLAB)

> **Looking for the BRSR dataset?** Download the official benchmark from Zenodo ([DOI 10.5281/zenodo.23010395](https://doi.org/10.5281/zenodo.23010395)) with `python download_data.py` in the repository root. The scripts in this folder were not seeded, so running them produces *new* BRSR-style data, not the published benchmark. The Python `DataPreparation_*.py` scripts are the original preparation step (MATLAB output → pickle) and are kept for reference; the released HDF5 files replace them.

This directory contains MATLAB scripts for generating baseline and extended radar signal datasets used in BRSR-OpGAN research.

## Files

### Generation Scripts
- **DataGeneration_Baseline.m** - Generates signals corrupted with AWGN at fixed SNR levels
- **DataGeneration_Extended.m** - Generates signals with randomized blends of multiple artifacts
- **add_distortion.m** - Helper function that applies AWGN, echo, and interference artifacts

### Visualization Tools
- **plot_signals_awgn.m** - Visualizes baseline signals (clean vs. AWGN-corrupted)
- **plot_signals_blind.m** - Visualizes extended dataset signals with multiple artifact components

### Data Preparation Scripts
- **DataPreparation_Baseline.py** - Processes baseline dataset and splits into train/validation/test sets
- **DataPreparation_Extended.py** - Processes extended dataset with multiple distortion types

### Required Data
- **mixing_signals.mat** - Contains signal patterns used for interference simulation
- **waveform-types/** - Directory containing radar waveform generation functions

## Generated Datasets

### Baseline Dataset
- **Corruption type:** AWGN only (Additive White Gaussian Noise)
- **SNR levels:** -14 dB to 10 dB (2 dB steps)
- **Training set:** 400 samples × 12 waveforms × 13 SNR levels = 62,400 signals
- **Test set:** 150 samples × 12 waveforms × 13 SNR levels = 23,400 signals

### Extended BRSR Dataset
- **Corruption types:** Random combination of AWGN, Echo, and Interference
- **SNR levels:** Random values between -14 dB and 10 dB
- **Training set:** 400 samples × 12 waveforms × 13 SNR levels = 62,400 signals
- **Test set:** 150 samples × 12 waveforms × 13 SNR levels = 23,400 signals

## Radar Waveform Types (12)

| Type  | Description |
|-------|-------------|
| LFM   | Linear Frequency Modulation |
| Costas | Frequency hopping |
| BPSK  | Binary Phase Shift Keying (Barker codes) |
| Frank | Polyphase code |
| P1-P4 | Polyphase codes |
| T1-T4 | Frequency shift keying variants |

## Output Format

Each dataset contains the following MATLAB data files:

### Baseline Dataset

```
Dataset_baseline_train/
├── X_clean_signals_train.mat  - Original radar signals [N×2×1024]
├── X_distorted_signals_train.mat - AWGN-corrupted signals [N×2×1024]
├── Y_label_train.mat - Waveform class labels [N×1]
└── Y_SNR_train.mat - SNR values [N×1]

Dataset_baseline_test/
├── X_clean_signals_test.mat
├── X_distorted_signals_test.mat
├── Y_label_test.mat
└── Y_SNR_test.mat
```

### Extended Dataset

```
Dataset_extended_train/
├── X_clean_signals_train.mat - Clean signals [N×2×1024]
├── X_distorted_signals_train.mat - Combined corrupted signals [N×2×1024]
├── X_distortion_awgn_train.mat - AWGN component [N×2×1024]
├── X_distortion_echo_train.mat - Echo component [N×2×1024]
├── X_distortion_cci_train.mat - Interference component [N×2×1024]
├── Y_label_train.mat - Waveform class labels [N×1]
└── Y_SNR_train.mat - SNR values [N×1]

Dataset_extended_test/
└── [Same structure as train]
```

## Usage

1. **Setup**
   ```matlab
   addpath 'waveform-types'
   ```

2. **Generate baseline dataset**
   ```matlab
   run DataGeneration_Baseline.m
   ```

3. **Generate extended dataset**
   ```matlab
   run DataGeneration_Extended.m
   ```

4. **Process datasets with Python**
   ```python
   # Process baseline dataset
   python DataPreparation_Baseline.py
   
   # Process extended dataset
   python DataPreparation_Extended.py
   ```

## Prepared Dataset Structure

After running the Python preparation scripts, the datasets are organized as follows:

```
Prepared_Dataset/
├── base_dataset.pickle - Contains baseline dataset
└── extended_dataset.pickle - Contains extended dataset with distortion components
```

Each pickle file contains a dictionary with the following structure:
```
{
    'train': {
        'clean': array[...],    # Clean signals (training set)
        'noisy': array[...],    # Noisy signals (training set)
        'label': array[...],    # Class labels (training set)
        'SNR': array[...],      # SNR values (training set)
        'distortions': array[...], # [For extended dataset only]
    },
    'validation': {
        'clean': array[...],    # Clean signals (validation set)
        'noisy': array[...],    # Noisy signals (validation set)
        'label': array[...],    # Class labels (validation set)
        'SNR': array[...],      # SNR values (validation set)
        'distortions': array[...], # [For extended dataset only]
    },
    'test': {
        'clean': array[...],    # Clean signals (test set)
        'noisy': array[...],    # Noisy signals (test set)
        'label': array[...],    # Class labels (test set)
        'SNR': array[...],      # SNR values (test set)
        'distortions': array[...], # [For extended dataset only]
    }
}
```

## Requirements

### MATLAB
- MATLAB R2019b or later
- Signal Processing Toolbox
- Time-Frequency Toolbox (tftb-0.2)

### Python
- Python 3.7+
- h5py
- numpy
- scikit-learn
- pickle

## Citation

If you use these datasets in your research, please cite our papers:

```bibtex
@article{zahid2024brsr,
  title={{BRSR-OpGAN}: Blind radar signal restoration using operational generative adversarial network},
  author={Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  journal={Neural Networks},
  volume={190},
  pages={107709},
  year={2025},
  doi={10.1016/j.neunet.2025.107709}
}
```

```bibtex
@article{zahid2025core,
  title={CoRe-Net: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration},
  author={Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2501.17125},
  year={2025}
}
```

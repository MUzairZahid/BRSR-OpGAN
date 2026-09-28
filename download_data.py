"""Download the BRSR benchmark dataset from Zenodo (DOI 10.5281/zenodo.23010395) and verify checksums.

Examples
--------
python download_data.py                              # everything (~3.2 GB)
python download_data.py --dataset brsr               # BRSR only (~2.1 GB)
python download_data.py --dataset brsr --splits test # only what evaluate.py needs
"""
import argparse
import hashlib
import os
import sys
import urllib.request

RECORD_ID = "23010395"
BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files/{{name}}?download=1"  # {name} is filled in
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, out_dir, base_url=BASE_URL):
    dst = os.path.join(out_dir, name)
    tmp = dst + ".part"
    with urllib.request.urlopen(base_url.format(name=name)) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                sys.stdout.write(f"\r  {name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    os.replace(tmp, dst)
    print()
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["brsr", "awgn_baseline", "all"], default="all")
    ap.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                    default=["train", "validation", "test"])
    ap.add_argument("--out_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--base_url", default=BASE_URL, help="download URL template (for mirrors)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = ["brsr", "awgn_baseline"] if args.dataset == "all" else [args.dataset]
    wanted = ["README.md", "interference_bank.h5"]
    for ds in datasets:
        wanted += [f"{ds}_{s}.h5" for s in args.splits] + [f"{ds}_metadata.csv"]

    sums_path = fetch("SHA256SUMS.txt", args.out_dir, args.base_url)
    expected = {}
    with open(sums_path) as f:
        for line in f:
            if line.strip():
                digest, name = line.split(maxsplit=1)
                expected[name.strip().lstrip("*./")] = digest

    for name in wanted:
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path) and sha256(path) == expected.get(name):
            print(f"  {name}: already present, checksum OK")
            continue
        fetch(name, args.out_dir, args.base_url)
        ok = sha256(path) == expected.get(name)
        print(f"  {name}: checksum {'OK' if ok else 'MISMATCH - please re-run'}")
        if not ok:
            sys.exit(1)
    print(f"Done. Files are in {args.out_dir}")


if __name__ == "__main__":
    main()

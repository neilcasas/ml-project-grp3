"""
SICAPv2 dataset setup helper.

The dataset must be downloaded manually from Mendeley Data:
  https://data.mendeley.com/datasets/9xxm58dvs3/

After downloading, extract so the layout is:
  <project_root>/SICAPv2/
      images/          <- 18,783 .jpg patches (512x512, 10X)
      masks/           <- pixel-level annotation masks
      partition/
          Test/        <- Test.xlsx, Train.xlsx, *Cribriform.xlsx
          Validation/
              Val1/ Val2/ Val3/ Val4/
      readme.txt
      wsi_labels.xlsx

Run this script from the project root to verify the layout is correct:
  python data/download_sicapv2.py [--root <path>]
"""

import argparse
import sys
from pathlib import Path


REQUIRED = [
    "images",
    "masks",
    "partition/Test/Test.xlsx",
    "partition/Test/Train.xlsx",
    "partition/Validation/Val1/Train.xlsx",
    "partition/Validation/Val1/Test.xlsx",
    "readme.txt",
]


def verify(root: Path) -> bool:
    ok = True
    for rel in REQUIRED:
        p = root / rel
        if not p.exists():
            print(f"  MISSING: {p}")
            ok = False
        else:
            print(f"  OK:      {p}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify SICAPv2 dataset layout.")
    parser.add_argument(
        "--root",
        default=None,
        help="Path to SICAPv2 root. Defaults to <project_root>/SICAPv2.",
    )
    args = parser.parse_args()

    if args.root:
        sicap_root = Path(args.root)
    else:
        project_root = Path(__file__).resolve().parent.parent
        sicap_root = project_root / "SICAPv2"

    print(f"Checking SICAPv2 at: {sicap_root}\n")

    if not sicap_root.exists():
        print("ERROR: SICAPv2 directory not found.")
        print(
            "\nPlease download from: https://data.mendeley.com/datasets/9xxm58dvs3/"
            "\nand extract to: <project_root>/SICAPv2/"
        )
        sys.exit(1)

    if verify(sicap_root):
        img_count = len(list((sicap_root / "images").glob("*.jpg")))
        print(f"\nSICAPv2 OK — {img_count} images found.")
    else:
        print("\nSome files are missing. Re-check your extraction.")
        sys.exit(1)


if __name__ == "__main__":
    main()

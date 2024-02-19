#!/usr/bin/env python3
import argparse
from pathlib import Path

from cached_opentims.io import dump_tdf_to_startrek

parser = argparse.ArgumentParser(
    description="Dump .tdf_raw dataset to mmapped_df `.startrek` format. Translate .tdf to parquet and save too."
)
parser.add_argument(
    "folder_d",
    help="Path to the .d folder",
    type=Path,
)
parser.add_argument(
    "output_startrek",
    help="Path to the output folder.",
    type=Path,
)
parser.add_argument(
    "--progressbar_message",
    help="Path to the output folder.",
    default="Dumping .tdf to .startrek",
)

args = parser.parse_args()


if __name__ == "__main__":
    assert not args.output_startrek.exists(), "Folder exists: it ain't allowed. act."

    dump_tdf_to_startrek(
        rawdata_path=args.folder_d,
        output_folder=args.output_startrek,
        _progressbar_message=args.progressbar_message,
    )

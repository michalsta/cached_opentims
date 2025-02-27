#!/usr/bin/env python3
import argparse
from pathlib import Path

from cached_opentims.io import create_and_open_cached_tdf


def main():
    parser = argparse.ArgumentParser(
        description="Dump .tdf_raw dataset to mmapped_df `.startrek` format. Translate .tdf to parquet and save too."
    )
    parser.add_argument(
        "folder_d",
        help="Path to the .d folder",
        type=Path,
    )
    parser.add_argument(
        "folder_startrek",
        help="Path to the output folder.",
        type=Path,
    )
    parser.add_argument(
        "--progressbar_message",
        help="Message to display in progressbar.",
        default="Dumping .tdf to .startrek",
    )

    args = parser.parse_args()

    create_and_open_cached_tdf(
        folder_d=args.folder_d,
        folder_startrek=args.folder_startrek,
        _progressbar_message=args.progressbar_message,
    )


if __name__ == "__main__":
    main()

from pathlib import Path

from tqdm import tqdm as progressbar

import mmapped_df
import numpy as np
import numpy.typing as npt
import opentimspy
import pandas as pd

from .stats import count2D


def dump_tdf_to_memmapped_numpy_file(
    path: str,
    raw_data_handler: opentimspy.OpenTIMS,
    progressbar_message: str = "Dumping to file.",
) -> npt.NDArray:
    events_cnt = raw_data_handler.frames["NumPeaks"].sum()
    sparse = np.memmap(path, dtype=np.uint32, mode="w+", shape=(events_cnt, 4))
    visited_events = 0
    for frame in progressbar(
        range(raw_data_handler.min_frame, raw_data_handler.max_frame + 1),
        desc=progressbar_message,
    ):
        raw = raw_data_handler[frame]
        sparse[visited_events : visited_events + len(raw)] = raw
        visited_events += len(raw)
    return sparse


def read_memmapped_events_file(path, mode="r"):
    return np.memmap("memmapped", dtype=np.uint32, mode=mode).reshape(-1, 4)


def dump_tdf_to_startrek(
    rawdata_path: Path,
    output_folder: Path,
    _progressbar_message: str = "Dumping raw data to `.startrek`.",
    _columns=[
        "frame",
        "scan",
        "tof",
        "intensity",
        "retention_time",
        "inv_ion_mobility",
        "mz",
    ],
    **kwargs,
) -> None:
    raw_data_handler = opentimspy.OpenTIMS(rawdata_path)
    startrek_handler = mmapped_df.DatasetWriter(output_folder, **kwargs)
    frame_scan_counts = np.zeros(
        dtype=np.uint32,
        shape=(raw_data_handler.max_frame + 1, raw_data_handler.max_scan + 1),
    )
    maxes = []
    for frame in progressbar(
        range(raw_data_handler.min_frame, raw_data_handler.max_frame + 1),
        desc=_progressbar_message,
    ):
        raw = raw_data_handler.query(frame, columns=_columns)
        raw = pd.DataFrame(raw, columns=_columns, copy=False)
        maxes.append({c: raw[c].max() for c in raw.columns})
        count2D(frame_scan_counts, raw.frame.to_numpy(), raw.scan.to_numpy())
        startrek_handler.append_df(raw)
    maxes = pd.DataFrame(maxes)
    maxes = pd.DataFrame([{c: maxes[c].max() for c in maxes.columns}])
    np.save(output_folder / "counts.npy", frame_scan_counts)
    maxes.to_parquet(output_folder / "maxes.npy")
    pd.DataFrame(raw_data_handler.frames).to_parquet(output_folder / "frames.parquet")


def read_startrek(
    startrek_path: Path,
) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray, pd.DataFrame]:
    df = mmapped_df.open_dataset(startrek_path)
    frame_scan_counts = np.load(startrek_path / "counts.npy")
    maxes = pd.read_parquet(startrek_path / "maxes.npy")
    frames = pd.read_parquet(startrek_path / "frames.parquet")
    return df, frame_scan_counts, maxes, frames

import typing
from pathlib import Path

from tqdm import tqdm as progressbar

import mmapped_df
import numba
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


@numba.njit
def index(frames, scans, starts, counts):
    for idx in range(len(frames)):
        frame = frames[idx]
        scan = scans[idx]
        if starts[frame, scan] == 0:
            starts[frame, scan] = idx
        counts[frame, scan] += 1

    last_frame = 0
    for frame in range(starts.shape[0]):
        for scan in range(starts.shape[1]):
            if starts[frame, scan] == 0:
                starts[frame, scan] = last_frame
            last_frame = starts[frame, scan]


def create_and_open_cached_tdf(
    folder_startrek: Path | str,
    folder_d: Path | str | None = None,
    _progressbar_message: str = "Dumping raw data to mmapped format.",
    _columns: list[str] = [
        "frame",
        "scan",
        "tof",
        "intensity",
        "mz",
        "inv_ion_mobility",
        "retention_time",
    ],
) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray, dict[str, typing.Any], pd.DataFrame]:
    """
    Dump raw tdf format into startrek format if necessary and then open it.
    Dump also basic statistics such as maxes, and tables mapping frame and scan to start (whatever it is) and counts of events.

    Arguments:
        folder_d (Path|str|None): Path to the .d folder with data in the tdf format. Might be neglected when the 'folder_startrek' exists.
        folder_startrek (Path|str): Path to the output folder in the startrek format.

    Returns:
        tuple: data frame with mmapped raw data, starts, counts, maxes, and the original Frames table from the 'analysis.tdf'.

    Raises:
        no concerns.
    """
    folder_startrek = Path(folder_startrek)

    _fresh = not folder_startrek.exists()
    if _fresh:
        assert folder_d is not None
        folder_d = Path(folder_d)
        raw_data_handler = opentimspy.OpenTIMS(folder_d)
        with mmapped_df.DatasetWriter(folder_startrek, overwrite_dir=True) as DW:
            for frame in progressbar(
                raw_data_handler.frames["Id"],
                desc=_progressbar_message,
            ):
                raw_frame = pd.DataFrame(
                    raw_data_handler.query(frame, columns=_columns),
                    columns=_columns,
                    copy=False,
                )
                DW.append_df(raw_frame)

    df = mmapped_df.open_dataset(folder_startrek)

    if _fresh:
        maxes = {c: np.max(df[c].to_numpy()) for c in df.columns}
        assert maxes["frame"] <= raw_data_handler.max_frame
        assert maxes["scan"] <= raw_data_handler.max_scan
        stats_shape = raw_data_handler.max_frame + 1, raw_data_handler.max_scan + 1
        pd.DataFrame([maxes]).to_parquet(folder_startrek / "maxes.parquet")

        starts = np.zeros(shape=stats_shape, dtype=np.uint64)
        counts = np.zeros(shape=stats_shape, dtype=np.uint64)
        index(
            df.frame.to_numpy(),
            df.scan.to_numpy(),
            starts,
            counts,
        )
        np.save(folder_startrek / "counts.npy", counts)
        np.save(folder_startrek / "starts.npy", starts)
        Frames = pd.DataFrame(raw_data_handler.frames)
        Frames.to_parquet(folder_startrek / "frames.parquet")
    else:
        counts = np.load(folder_startrek / "counts.npy")
        starts = np.load(folder_startrek / "starts.npy")
        maxes = pd.read_parquet(folder_startrek / "maxes.parquet").to_dict(
            orient="records"
        )[0]
        Frames = pd.read_parquet(folder_startrek / "frames.parquet")

    return df, starts, counts, maxes, Frames

import math

import numba
import numpy as np
import numpy.typing as npt

from .misc import assign


@numba.njit
def minmax(xx, _min=math.inf, _max=-math.inf):
    for x in xx:
        _min = min(x, _min)
        _max = max(x, _max)
    _min, _max = np.array([_min, _max], dtype=xx.dtype)
    return _min, _max


@numba.njit
def count_nonzero(neighborhood):
    return np.uint32(np.count_nonzero(neighborhood))


@numba.njit
def count1D(counts: npt.NDArray, xx: npt.NDArray) -> None:
    assert len(counts.shape) == 1
    for i in range(len(xx)):
        counts[xx[i]] += 1


@numba.njit
def count2D(counts: npt.NDArray, xx: npt.NDArray, yy: npt.NDArray) -> None:
    assert len(counts.shape) == 2
    assert len(xx) == len(yy)
    for i in range(len(xx)):
        counts[xx[i], yy[i]] += 1


@numba.njit
def count3D(
    counts: npt.NDArray, xx: npt.NDArray, yy: npt.NDArray, zz: npt.NDArray
) -> None:
    assert len(counts.shape) == 3
    assert len(xx) == len(yy)
    assert len(xx) == len(zz)
    for i in range(len(xx)):
        counts[xx[i], yy[i], zz[i]] += 1


def cumsum(xx: npt.NDArray, inplace: bool = False) -> npt.NDArray:
    return np.cumsum(
        xx,
        out=xx.reshape(np.prod(xx.shape)) if inplace else None,
    ).reshape(xx.shape)


def counts_to_cumsum_idx(counts: npt.NDArray) -> npt.NDArray:
    """Counts to cumsums.

    Arguments:
        counts (npt.NDArray): tensor of any dimension.

    Returns:
        npt.NDArray: tensor with the input dimension plus one, containing C-order cumsums @1 and index-lagged cumsums @0.
    """
    long_cumsums = np.cumsum(counts.flatten())
    cumsum_idx = np.zeros(
        shape=counts.shape + (2,),
        dtype=counts.dtype,
    )
    curr = cumsum_idx[..., 1].reshape(np.prod(counts.shape))
    assign(curr, long_cumsums)
    prev = cumsum_idx[..., 0].reshape(np.prod(counts.shape))
    assign(prev[1:], long_cumsums[:-1])
    assert np.all(cumsum_idx[..., 0] <= cumsum_idx[..., 1])
    return cumsum_idx

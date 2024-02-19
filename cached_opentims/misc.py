import math

import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def is_nondecreasing(xx) -> bool:
    x_prev = -math.inf
    for x in xx:
        if x < x_prev:
            return False
    return True


@numba.njit
def is_sorted(xx) -> bool:
    prev_x = -math.inf
    for x in xx:
        if x <= prev_x:
            return False
        prev_x = x
    return True


@numba.njit
def round_up(x, digit):
    x_rnd = round(x, digit)
    if x_rnd < x:
        return x_rnd + 10 ** (-digit)
    return x_rnd


@numba.njit
def assign(xx, yy):
    assert len(xx) == len(yy)
    for i in range(len(xx)):
        xx[i] = yy[i]


@numba.njit
def expand_left_right_indices(left_right_idxs) -> npt.NDArray:
    cnt = 0
    for left_idx, right_idx in left_right_idxs:
        cnt += right_idx - left_idx
    expanded = np.empty(shape=(cnt,), dtype=left_right_idxs.dtype)
    j = 0
    for left_idx, right_idx in left_right_idxs:
        for idx in range(left_idx, right_idx):
            expanded[j] = idx
            j += 1
    return expanded

import math

import numba


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

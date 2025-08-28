import numpy as np

def unpack_array(L, n=3):
    """
    Unpack the last axis of length `n` into separate arrays.

    Parameters
    ----------
    L : array-like, shape (..., n)
        Input array whose last axis has length `n`. Does NOT support transposed (n, N).
    n : int, default 3
        Number of components to unpack from the last axis.

    Returns
    -------
    tuple
        n arrays with shape (...). If L is a single n-vector (shape (n,)),
        returns n Python scalars for backward compatibility.
    """
    A = np.asarray(L, dtype=float)

    if A.shape == ():  # scalar
        raise ValueError(f"Expected an array with last axis {n}; got scalar {A!r}")
    if A.shape[-1] != n:
        raise ValueError(f"Expected last axis length {n} (…, {n}); got shape {A.shape}.")

    comps = tuple(A[..., i] for i in range(n))

    # Preserve scalar returns for a single n-vector
    if comps[0].ndim == 0:
        return tuple(float(c) for c in comps)
    return comps

def interval_indices(x, q, *, clamp=False):
    """
    Return left/right indices i, i+1 for each q so that x[i] <= q < x[i+1].
    If clamp=True, out-of-bounds q are clamped to the nearest valid interval.
    """
    x = np.asarray(x, float)
    q = np.asarray(q, float)

    i = np.searchsorted(x, q, side="right") - 1
    if clamp:
        i = np.clip(i, 0, len(x) - 2)
    else:
        oob = (q < x[0]) | (q > x[-1])
        if np.any(oob):
            raise ValueError("Query outside [x[0], x[-1]].")

    return i, i + 1

def interval_with_fraction(x, q, *, clamp=False):
    """
    Also returns interpolation fraction t in [0,1]:
      q = (1-t)*x[i] + t*x[i+1]
    """
    i, j = interval_indices(x, q, clamp=clamp)
    x = np.asarray(x, float)
    q = np.asarray(q, float)
    t = (q - x[i]) / (x[j] - x[i])
    return i, j, t

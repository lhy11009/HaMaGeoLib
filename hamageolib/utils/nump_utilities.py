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

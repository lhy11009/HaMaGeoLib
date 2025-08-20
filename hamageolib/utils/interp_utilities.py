import numpy as np
from scipy.spatial import cKDTree

class KNNInterpolatorND:
    """
    kNN interpolation for scattered ND data with NaN filtering.

    Parameters
    ----------
    X : array-like, shape (n_samples, d) or list/tuple of d arrays broadcastable to (n_samples,)
        Sample coordinates.
    Z : array-like, shape (n_samples,)
        Sample values (NaNs ignored).
    k : int, default 3
        Number of neighbors (uses all if k > n).
    weights : {"distance","uniform"} or callable, default "distance"
        Weighting mode. If callable, it receives distances array of shape (n_queries, k).
    p : float, default 2
        Power for inverse-distance weighting; ignored for uniform or custom.
    scale : float or array-like of length d, default None
        Per-dimension scale factors applied to X and queries before distance calcs
        (useful when coordinates have different units or anisotropy).
    leafsize : int, default 16
        KD-tree leaf size.
    """
    def __init__(self, X, Z, k=3, weights="distance", p=2, scale=None, leafsize=16,\
                 max_distance=np.inf):
        self.max_distance = float(max_distance)
        X, Z = self._coerce_XZ(X, Z)
        mask = ~np.isnan(Z)
        if not np.any(mask):
            raise ValueError("All Z values are NaN; nothing to interpolate.")

        self.X = X[mask]
        self.Z = Z[mask]
        self.n, self.d = self.X.shape
        self.k = int(k)
        self.p = float(p)
        self.leafsize = int(leafsize)

        if callable(weights):
            self.weight_fn = weights
            self._mode = "callable"
        else:
            if weights not in ("distance", "uniform"):
                raise ValueError("weights must be 'distance', 'uniform', or a callable")
            self._mode = weights
            self.weight_fn = None

        # set up scaling
        if scale is None:
            self.scale = np.ones(self.d, dtype=float)
        else:
            self.scale = np.broadcast_to(np.asarray(scale, dtype=float), (self.d,))
        Xs = self.X / self.scale  # scaled coords for the tree
        self.tree = cKDTree(Xs, leafsize=self.leafsize)

    def __call__(self, *coords, fill_value=np.nan):
        Q, out_shape = self._coerce_queries(*coords)  # shape (m, d)
        if Q.size == 0:
            return np.empty(out_shape, dtype=float)

        Qs = Q / self.scale
        k_eff = min(self.k, self.n)
        d, idx = self.tree.query(Qs, k=k_eff, workers=-1, distance_upper_bound=self.max_distance)
        # d = np.atleast_2d(d)
        # idx = np.atleast_2d(idx)

        # Make sure first dim is m (num queries) and second dim is k (neighbors)
        d = np.asarray(d)
        idx = np.asarray(idx)
        if d.ndim == 1:               # happens when k_eff == 1
            d = d.reshape(-1, 1)
            idx = idx.reshape(-1, 1)
        else:
            # For safety, ensure (m, k_eff) not (k_eff, m)
            d = d.reshape(-1, k_eff)
            idx = idx.reshape(-1, k_eff)

        out = np.full(d.shape[0], np.nan, dtype=float)

        # Mark valid neighbors: finite distance AND index in-bounds
        valid_nbr = np.isfinite(d) & (idx < self.n)

        # If no valid neighbor for a query -> remain fill_value
        any_valid = valid_nbr.any(axis=1)
        if not np.any(any_valid):
            return out.reshape(out_shape)

        # Exact matches only if the first neighbor is valid and at zero distance
        exact = any_valid & valid_nbr[:, 0] & (d[:, 0] == 0.0)
        out[exact] = self.Z[idx[exact, 0]]

        need = ~exact & any_valid
        if np.any(need):
            di = d[need]                  # (r, k)
            ii = idx[need]                # (r, k)
            vm = valid_nbr[need]          # (r, k)

            # Safe indices: replace invalid with 0, and zero their weights later
            ii_safe = ii.copy()
            ii_safe[~vm] = 0
            zi = self.Z[ii_safe]          # safe: no OOB

            # Weights: zero out invalid neighbors
            if self._mode == "uniform":
                w = vm.astype(float)
            elif self._mode == "distance":
                w = np.zeros_like(di, dtype=float)
                w[vm] = 1.0 / np.maximum(di[vm], 1e-15) ** self.p
            else:  # callable
                w = np.asarray(self.weight_fn(di), dtype=float)
                if w.shape != di.shape:
                    raise ValueError("Custom weight function must return same shape as distances")
                w[~vm] = 0.0

            wsum = w.sum(axis=1)
            bad = (wsum == 0.0)  # no valid neighbors within cutoff
            # Normalize where we have at least one valid neighbor
            w[~bad] = w[~bad] / wsum[~bad, None]

            vals = np.where(bad, fill_value, np.sum(w * zi, axis=1))
            out[need] = vals

        return out.reshape(out_shape)

    # ---------- helpers ----------
    @staticmethod
    def _coerce_XZ(X, Z):
        Z = np.asarray(Z).ravel()
        if isinstance(X, (list, tuple)):
            cols = [np.asarray(c).ravel() for c in X]
            X = np.column_stack(cols)
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        if X.shape[0] != Z.shape[0]:
            raise ValueError("X and Z must have the same number of samples")
        return X.astype(float), Z.astype(float)

    def _coerce_queries(self, *coords):
        if len(coords) == 1:
            Q = np.asarray(coords[0])
            if Q.ndim == 1:
                # either a single point of length d, or m points in 1D
                if Q.shape[0] == self.d:
                    Q2 = Q.reshape(1, self.d)
                    return Q2, ()
                else:
                    if self.d != 1:
                        raise ValueError("For 1D query vector, interpolator must be 1D")
                    return Q.reshape(-1, 1), (Q.shape[0],)
            elif Q.ndim >= 2:
                if Q.shape[-1] != self.d:
                    raise ValueError(f"Last dim of query must be {self.d}")
                out_shape = Q.shape[:-1]
                return Q.reshape(-1, self.d), out_shape
        else:
            # coords are d separate arrays
            if len(coords) != self.d:
                raise ValueError(f"Expected {self.d} coordinate arrays, got {len(coords)}")
            shapes = [np.asarray(c).shape for c in coords]
            if not all(s == shapes[0] for s in shapes):
                raise ValueError("All coordinate arrays must have the same shape")
            Q = np.column_stack([np.asarray(c).ravel() for c in coords])
            return Q, shapes[0]
        raise ValueError("Could not parse query coordinates")

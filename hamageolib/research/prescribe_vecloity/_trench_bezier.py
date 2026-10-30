"""WorldBuilder-compatible cubic-Bezier trench sampling.

Control-handle rules ported from WorldBuilder's objects/bezier_curve.cc
(indicator_field, 69e72be49). Copyright (C) 2018-2026 WorldBuilder authors.
This module is distributed under LGPL version 2 or later, as is that source.
See WORLD_BUILDER_COPYING and WORLD_BUILDER_COPYING.LESSER.
"""
import math

import numpy as np


def sample_trench(coordinates, samples_per_segment=20):
    """Return longitude, latitude and finite-difference tangent in radians."""
    p = np.radians(np.asarray(coordinates, dtype=float))
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('Trench requires at least three finite lon/lat points')
    if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) == 0):
        raise ValueError('Consecutive trench coordinates must be distinct')
    n = len(p)
    angles = np.empty(n)
    d = p[1] - p[0]
    angles[0] = math.atan2(d[1], d[0])
    d = p[-2] - p[-1]
    angles[-1] = math.atan2(d[1], d[0])
    for i in range(1, n - 1):
        before, after = p[i - 1] - p[i], p[i + 1] - p[i]
        angles[i] = (math.atan2(before[1], before[0])
                     + math.atan2(after[1], after[0])) * .5 - math.pi * .5
    control = np.empty((n - 1, 2, 2))

    def handle(origin, angle, length):
        return origin + length * .2 * np.array([math.cos(angle), math.sin(angle)])

    def side(a, b, q):
        return (a[0] - b[0]) * (q[1] - a[1]) - (a[1] - b[1]) * (q[0] - a[0]) < 0

    length = np.linalg.norm(p[0] - p[1])
    control[0, 0] = handle(p[0], angles[0], length)
    control[0, 1] = handle(p[1], angles[1], length)
    if side(p[0], p[1], control[0, 1]) == side(p[0], p[1], p[2]):
        control[0, 1] = handle(p[1], angles[1] + math.pi, length)
    for i in range(1, n - 1):
        a, b = p[i], p[i + 1]
        length = np.linalg.norm(a - b)
        control[i, 0] = handle(a, angles[i], length)
        if side(a, b, control[i - 1, 1]) == side(a, b, control[i, 0]):
            control[i, 0] = handle(a, angles[i] + math.pi, length)
        control[i, 1] = handle(b, angles[i + 1], length)
        if i + 1 < n - 1 and side(a, b, control[i, 1]) == side(a, b, p[i + 2]):
            control[i, 1] = handle(b, angles[i + 1] + math.pi, length)

    def evaluate(i, t):
        u = 1 - t
        return (u*u*u*p[i] + 3*u*u*t*control[i, 0]
                + 3*t*t*u*control[i, 1] + t*t*t*p[i + 1])

    rows = []
    for i in range(n - 1):
        for j in range(samples_per_segment + 1):
            if i and j == 0:
                continue
            t = j / samples_per_segment
            point = evaluate(i, t)
            tangent = evaluate(i, min(1., t + 1e-5)) - evaluate(i, max(0., t - 1e-5))
            rows.append([*point, *tangent])
    return np.asarray(rows)

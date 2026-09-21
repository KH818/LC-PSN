"""Decode a spectrum into separated DOA peaks using a shared deterministic rule."""

import math

import numpy as np


def separated_topk(score, khat, grid, min_sep_deg=2.0):
    """Decode top-K separated angles from a spectrum.

    Interior local maxima are ranked first. If too few separated local maxima are
    available, the full grid is ranked and used as backfill until K angles are
    returned whenever the grid permits it.
    """
    score = np.asarray(score, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    khat = int(khat)
    min_sep = math.radians(float(min_sep_deg))

    if khat <= 0 or score.size == 0:
        return np.asarray([], dtype=np.float64)
    if score.shape[0] != grid.shape[0]:
        raise ValueError("score and grid must have the same length")

    peak_mask = np.zeros(score.shape[0], dtype=bool)
    if score.shape[0] >= 3:
        peak_mask[1:-1] = (score[1:-1] > score[:-2]) & (score[1:-1] > score[2:])
    peak_idx = np.nonzero(peak_mask)[0]

    def ranked(indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return indices
        order = np.lexsort((indices, -score[indices]))
        return indices[order]

    selected = []

    def accept(idx):
        return all(abs(grid[int(idx)] - grid[j]) >= min_sep for j in selected)

    for idx in ranked(peak_idx):
        if accept(idx):
            selected.append(int(idx))
        if len(selected) >= khat:
            break

    if len(selected) < khat:
        for idx in ranked(np.arange(score.shape[0])):
            if int(idx) in selected:
                continue
            if accept(idx):
                selected.append(int(idx))
            if len(selected) >= khat:
                break

    selected = sorted(selected[:khat])
    return np.asarray([grid[i] for i in selected], dtype=np.float64)

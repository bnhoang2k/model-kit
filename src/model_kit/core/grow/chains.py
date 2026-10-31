"""Limbs as continuation chains: one rule, shared by the grower and the measured-tree converter.

A tree is a set of segments, each with a parent segment. A limb is a chain of segments that, at
every fork, runs on through the child with the largest base radius. The pipe model makes that
child the one that carries the most leaf area, so the chain follows the main wood.
"""

from __future__ import annotations

import numpy as np


def continuation_chains(
    parent: np.ndarray, base_radius: np.ndarray, angle: np.ndarray
) -> list[np.ndarray]:
    """Segments in, chains out.

    `parent[i]` is the segment `i` grows from, or -1 for a segment that stands on nothing.
    `base_radius[i]` is segment `i`'s radius where it starts, and `angle[i]` its angle from the
    parent's direction, in any one unit. At a fork the chain runs on through the child with the
    largest base radius; a tie goes to the smaller angle, then to the lower index. Each chain is
    an int array of segment indexes from base to tip. Chains are ordered so a chain's parent
    chain comes first. Raises ValueError when a segment cannot be reached from a root (a cycle).
    """
    parent = np.asarray(parent, dtype=int)
    base_radius = np.asarray(base_radius, dtype=float)
    angle = np.asarray(angle, dtype=float)
    n = len(parent)
    if not (len(base_radius) == len(angle) == n):
        raise ValueError("parent, base_radius and angle must have one entry per segment")
    if n == 0:
        return []
    if np.any(parent < -1) or np.any(parent >= n):
        raise ValueError("parent must be -1 or a segment index")

    # Children grouped by parent, best first: thickest, then straightest, then lowest index.
    order = np.lexsort((np.arange(n), angle, -base_radius, parent))
    kids = order[parent[order] >= 0]
    cut = np.searchsorted(parent[kids], np.arange(n + 1))  # kids of i: kids[cut[i]:cut[i + 1]]

    out: list[np.ndarray] = []
    heads = list(np.flatnonzero(parent < 0))
    seen = 0
    k = 0
    while k < len(heads):  # heads queue in parent-first order, so a parent chain is cut first
        chain = [int(heads[k])]
        k += 1
        while True:
            i = chain[-1]
            best = kids[cut[i]:cut[i + 1]]
            if not len(best):
                break
            chain.append(int(best[0]))
            heads.extend(int(j) for j in best[1:])
        seen += len(chain)
        out.append(np.array(chain, dtype=int))
    if seen != n:
        raise ValueError(f"{n - seen} segments cannot be reached from a root: the parents "
                         "hold a cycle")
    return out

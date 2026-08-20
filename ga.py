"""
ga.py -- a small real-coded genetic optimizer, deliberately model-agnostic.

The core hands it a loss over a bounded real vector; it never knows whether that
vector is a single leaf, one branch, or the full tree. That is the whole point:
"train a subtree" and "train the full tree" are the SAME call with a different
loss and different bounds, so the curriculum (branch fits -> full-tree fit) needs
no special optimizer code.

WARM START. `seeds=` injects known-good genomes into the initial population
(padded/clipped to the current bounds). For the full-tree fit we seed it with the
genomes identified on the single-fault branches, so the GA starts near a sensible
assembly instead of cold random -- it is a COMPUTE SAVER, not a constraint: the
seeded genes are free to move like any other.

Operators: tournament selection, BLX-alpha crossover, Gaussian mutation scaled to
each gene's range, elitism. Nothing exotic -- the fitness landscape here is mild
and the value is reproducibility, not cleverness.
"""

from __future__ import annotations
import numpy as np


def optimize(loss, lo, hi, *, gens=60, pop=60, seed=0, seeds=None,
             elite=2, tourn=3, alpha=0.3, p_mut=0.15, mut_scale=0.1,
             patience=None, verbose=False):
    """Minimize `loss(genome)->float` over the box [lo, hi].

    seeds : optional list of genomes to inject into gen 0 (warm start). Each is
            clipped to [lo,hi]; a seed shorter/longer than the genome is
            padded/truncated, so a branch genome can seed a full-tree slot.
    Returns (best_genome, best_loss, history_of_best_loss).
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    d = len(lo)
    span = np.where(hi > lo, hi - lo, 1.0)

    # ---- initial population -------------------------------------------------
    P = lo + rng.random((pop, d)) * span
    if seeds:
        for i, s in enumerate(seeds[:pop]):
            g = np.asarray(s, float)
            v = np.empty(d)
            m = min(d, len(g))
            v[:m] = g[:m]
            if m < d:                                   # pad short seed randomly
                v[m:] = lo[m:] + rng.random(d - m) * span[m:]
            P[i] = np.clip(v, lo, hi)

    def score(pop_):
        return np.array([loss(g) for g in pop_])

    F = score(P)
    order = np.argsort(F)
    P, F = P[order], F[order]
    best_g, best_f = P[0].copy(), float(F[0])
    hist = [best_f]

    stall = 0
    for _ in range(gens):
        nxt = [P[i].copy() for i in range(min(elite, pop))]   # elitism
        while len(nxt) < pop:
            a = _tournament(P, F, tourn, rng)
            b = _tournament(P, F, tourn, rng)
            c = _blx(a, b, alpha, rng)                         # crossover
            c = _mutate(c, lo, hi, span, p_mut, mut_scale, rng)  # mutation
            nxt.append(np.clip(c, lo, hi))
        P = np.array(nxt)
        F = score(P)
        order = np.argsort(F)
        P, F = P[order], F[order]
        if F[0] < best_f - 1e-12:
            best_g, best_f, stall = P[0].copy(), float(F[0]), 0
        else:
            stall += 1
        hist.append(best_f)
        if verbose:
            print(f"    gen {len(hist)-1:3d}  best={best_f:.5f}")
        if patience and stall >= patience:
            break
    return best_g, best_f, hist


def _tournament(P, F, k, rng):
    idx = rng.integers(0, len(P), size=k)
    return P[idx[np.argmin(F[idx])]]


def _blx(a, b, alpha, rng):
    lo_ = np.minimum(a, b)
    hi_ = np.maximum(a, b)
    d = hi_ - lo_
    return rng.uniform(lo_ - alpha * d, hi_ + alpha * d)


def _mutate(g, lo, hi, span, p, scale, rng):
    mask = rng.random(len(g)) < p
    g = g.copy()
    g[mask] += rng.normal(0.0, scale, mask.sum()) * span[mask]
    return g


def _self_test():
    # smooth bowl: optimum is exactly x = 0.3, so a seed there IS optimal.
    def loss(x):
        return float(np.sum((x - 0.3) ** 2))
    lo, hi = np.full(8, -1.0), np.full(8, 1.0)
    g_cold, f_cold, _ = optimize(loss, lo, hi, gens=80, pop=60, seed=1)
    good = np.full(8, 0.3)
    g_warm, f_warm, hist_warm = optimize(loss, lo, hi, gens=80, pop=60, seed=1,
                                         seeds=[good])
    print(f"cold best loss = {f_cold:.5f}   (x ~ {np.round(g_cold,2)})")
    print(f"warm best loss = {f_warm:.5f}   gen0 = {hist_warm[0]:.2e} (seeded at optimum)")
    assert f_cold < 1e-2, f_cold
    assert hist_warm[0] < 1e-9, "elitism must keep the optimal seed at gen 0"
    assert f_warm <= 1e-9, f_warm
    # short-seed padding must not crash (a branch genome seeding a bigger slot)
    optimize(loss, lo, hi, gens=5, pop=20, seed=0, seeds=[np.zeros(3)])
    print("ga self-test OK")


if __name__ == "__main__":
    _self_test()
"""_mock_evo2.py — a faithful NUMPY miniature of Evo 2's residual stream + readout.

Purpose: let wcl04/wcl07/wcl08 run END-TO-END with `--mock` (no GPU, no torch, no
weights) so their analysis code can be verified. The mock is NOT evidence — its
numbers are for CODE VERIFICATION ONLY. It reproduces the three architectural
properties the experiments depend on, so that "the code runs and returns sensible,
correctly-shaped, finite results" is a meaningful check:

  1. Residual accumulation: ‖h_ell‖ grows ~monotonically with depth (pre-norm
     residual adds), so a magnitude lens is depth-dominated — as in Evo 2.
  2. Scale-invariant readout: the next-token distribution is unembed(RMSNorm(h)),
     and RMSNorm(alpha*h) == RMSNorm(h) exactly, so the model's OUTPUT is provably
     blind to residual magnitude. (This is the crux of Experiment 7A and it is a
     real, exact property, not a rigged one.)
  3. Directional convergence: each token's direction drifts toward a token-specific
     target, so cosine-to-h_norm settling is well-defined and JSD->0 as direction
     converges — while trajectory smoothness stabilises on a partly-independent
     schedule (so the cos-vs-geo differential in Exp 7 is a genuine, non-forced
     numerical outcome of the dynamics, not hard-coded).

The REAL scripts import the identical analysis functions and feed them arrays from
Evo 2 via gDTR; only the array SOURCE differs (mock numpy vs. torch forward).
"""
from __future__ import annotations

import numpy as np

MOCK_N_LAYERS = 32
MOCK_HIDDEN = 64
MOCK_VOCAB = 16
EPS = 1e-6


def _rmsnorm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """RMSNorm: x / rms(x) * gamma. Exactly scale-invariant in x."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS)
    return x / rms * gamma


class MockEvo2:
    """Numpy mini-Evo2. Deterministic given seed."""

    def __init__(self, n_layers=MOCK_N_LAYERS, hidden=MOCK_HIDDEN, vocab=MOCK_VOCAB, seed=0):
        self.L, self.H, self.V = n_layers, hidden, vocab
        rng = np.random.default_rng(seed)
        self.gamma = rng.normal(1.0, 0.1, hidden)                 # RMSNorm learned gain
        self.W_unembed = rng.standard_normal((vocab, hidden)) / np.sqrt(hidden)
        self.embed = rng.standard_normal((256, hidden)) * 0.3     # token -> vector
        # per-layer gains that make direction converge and steps stabilise
        self.dir_gain = np.linspace(0.05, 0.9, n_layers)
        self.step_scale = np.linspace(1.0, 0.2, n_layers)

    # ---- forward that produces the residual-stream stack + post-norm state ----
    def forward_hidden(self, token_ids: np.ndarray, patch=None):
        """token_ids [T] int. Optional patch=(layer, pos, vector) overwrites the
        residual stream at that (layer, position) and propagates downstream.
        Returns h_stack [L, T, H] and h_norm [T, H]."""
        T = len(token_ids)
        rng = np.random.default_rng(int(token_ids.sum()) + 12345)
        targets = rng.standard_normal((T, self.H))
        targets /= np.linalg.norm(targets, axis=-1, keepdims=True) + EPS
        # per-token convergence rate + growth rate (so settling layers vary across
        # tokens and each lens has a non-constant, defined distribution)
        tok_rate = rng.uniform(0.6, 1.4, (T, 1))
        tok_grow = rng.uniform(0.05, 0.15, (T, 1))
        h = self.embed[token_ids % 256].copy()                    # [T, H]
        stack = np.zeros((self.L, T, self.H))
        for ell in range(self.L):
            stack[ell] = h
            if patch is not None and patch[0] == ell:
                _, pos, vec = patch
                stack[ell, pos] = vec
                h = stack[ell].copy()
            # pre-norm residual block: drift toward target + accumulate magnitude
            n = np.linalg.norm(h, axis=-1, keepdims=True) + EPS
            drift = (self.dir_gain[ell] * tok_rate) * (targets - h / n)
            noise = self.step_scale[ell] * 0.15 * rng.standard_normal((T, self.H))
            h = h + drift + noise + tok_grow * h                  # magnitude grows (per-token rate)
        h_norm = _rmsnorm(h, self.gamma)                          # post-final-norm state
        return stack, h_norm

    # ---- readout: scale-invariant by construction ----
    def decode(self, h: np.ndarray, is_post_norm: bool) -> np.ndarray:
        """logits [.., V]. is_post_norm=True: h already normed (h_norm tap).
        is_post_norm=False: apply RMSNorm first (per-layer logit lens)."""
        x = h if is_post_norm else _rmsnorm(h, self.gamma)
        return x @ self.W_unembed.T


# ---------- gDTR-API-compatible shims (so --mock swaps in cleanly) ----------

def load_evo2(**kw):
    return MockEvo2(seed=kw.get("seed", 0))


def tokenize(seq: str, bundle, device=None):
    return np.array([ord(c) for c in seq], dtype=np.int64)


def all_layer_names(n_layers=MOCK_N_LAYERS, include_norm=True):
    names = [f"blocks.{i}" for i in range(n_layers)]
    return names + (["norm"] if include_norm else [])


def extract_hidden_states(bundle, input_ids, save_layers=None, patch=None):
    """Return {blocks.i: [T,H], norm: [T,H]} as numpy (mock analogue of gDTR's)."""
    stack, h_norm = bundle.forward_hidden(np.asarray(input_ids), patch=patch)
    out = {f"blocks.{i}": stack[i] for i in range(bundle.L)}
    out["norm"] = h_norm
    return out


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def jsd_lens_np(h_stack: np.ndarray, h_norm: np.ndarray, bundle) -> np.ndarray:
    """Per-layer Jensen-Shannon distance between layer-ell next-token distribution
    and the FINAL distribution, normalised by log(V). Faithful numpy port of
    gDTR src/logit_lens_evo2.py::jsd_lens (D[L-1]:=0 convention). Returns [L, T]."""
    L, T, _ = h_stack.shape
    p_final = softmax(bundle.decode(h_norm, is_post_norm=True), axis=-1)          # [T,V]
    logV = np.log(bundle.V)
    D = np.zeros((L, T))
    for ell in range(L):
        p = softmax(bundle.decode(h_stack[ell], is_post_norm=False), axis=-1)     # [T,V]
        m = 0.5 * (p + p_final)
        kl_pm = (p * (np.log(p + 1e-30) - np.log(m + 1e-30))).sum(-1)
        kl_fm = (p_final * (np.log(p_final + 1e-30) - np.log(m + 1e-30))).sum(-1)
        D[ell] = np.clip(0.5 * (kl_pm + kl_fm), 0, None) / logV
    D[L - 1] = 0.0
    return D

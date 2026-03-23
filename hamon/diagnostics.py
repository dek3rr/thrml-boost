"""Diagnostics for sample quality and model health."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Shaped

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample convergence
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceReport:
    """Result of :func:`sample_convergence`.

    Attributes:
        status: ``"CONVERGED"``, ``"BORDERLINE"``, or ``"NEED_MORE"``.
        drifts: L1 drift in marginals between consecutive quartile checkpoints
            (three values: 25→50 %, 50→75 %, 75→100 %).
        rank_stability: Jaccard similarity of the top-*k* variables between
            the first and second half of the samples.
    """

    status: str
    drifts: list[float]
    rank_stability: float


def sample_convergence(
    samples: Bool[Array, "n_samples n_variables"],
    *,
    target_k: int = 15,
    drift_threshold: float = 0.01,
    jaccard_threshold: float = 0.8,
) -> ConvergenceReport:
    """Measure stability of marginal probability estimates.

    Splits *samples* into quartile checkpoints (25 %, 50 %, 75 %, 100 %),
    computes marginals at each checkpoint, and reports the L1 drift between
    consecutive checkpoints together with the rank stability of the top-*k*
    variables.

    Args:
        samples: boolean array of shape ``(n_samples, n_variables)``.
        target_k: number of top variables to track for rank stability.
        drift_threshold: maximum acceptable L1 drift per variable for the
            final checkpoint to be considered converged.
        jaccard_threshold: minimum Jaccard similarity of top-*k* sets
            between halves for rank stability to be considered converged.

    Returns:
        A :class:`ConvergenceReport`.
    """
    samples = jnp.asarray(samples)
    n_samples, n_vars = samples.shape
    target_k = min(target_k, n_vars)

    quartile_indices = [n_samples * q // 4 for q in range(1, 5)]
    marginals = [
        jnp.mean(samples[:idx].astype(jnp.float32), axis=0) for idx in quartile_indices
    ]

    drifts = [
        float(jnp.mean(jnp.abs(marginals[i + 1] - marginals[i])))
        for i in range(len(marginals) - 1)
    ]

    # Rank stability: Jaccard of top-k between first and second half.
    half = n_samples // 2
    m_first = jnp.mean(samples[:half].astype(jnp.float32), axis=0)
    m_second = jnp.mean(samples[half:].astype(jnp.float32), axis=0)

    top_first = set(jnp.argsort(-m_first)[:target_k].tolist())
    top_second = set(jnp.argsort(-m_second)[:target_k].tolist())
    jaccard = len(top_first & top_second) / len(top_first | top_second)

    final_drift = drifts[-1]
    if final_drift <= drift_threshold and jaccard >= jaccard_threshold:
        status = "CONVERGED"
    elif final_drift <= drift_threshold * 3 and jaccard >= jaccard_threshold * 0.8:
        status = "BORDERLINE"
    else:
        status = "NEED_MORE"

    return ConvergenceReport(status=status, drifts=drifts, rank_stability=jaccard)


# ---------------------------------------------------------------------------
# Marginal entropy
# ---------------------------------------------------------------------------


def marginal_entropy(
    samples: Bool[Array, "n_samples n_variables"],
) -> float:
    """Normalized entropy of the empirical marginal distribution.

    Computes the mean per-variable binary entropy, normalized to [0, 1].
    A value near 0 means most variables are frozen (all True or all False);
    near 1 means each variable is near 50/50.

    Args:
        samples: boolean array of shape ``(n_samples, n_variables)``.

    Returns:
        Scalar in [0, 1].
    """
    p = jnp.mean(jnp.asarray(samples).astype(jnp.float32), axis=0)
    # Use jnp.where to handle p=0 and p=1 without NaN from 0*log(0).
    safe_p = jnp.clip(p, 1e-10, 1.0 - 1e-10)
    h = -(safe_p * jnp.log2(safe_p) + (1 - safe_p) * jnp.log2(1 - safe_p))
    # Zero out entropy for variables that are truly frozen.
    h = jnp.where((p == 0.0) | (p == 1.0), 0.0, h)
    return float(jnp.mean(h))


# ---------------------------------------------------------------------------
# Energy balance
# ---------------------------------------------------------------------------


@dataclass
class EnergyBalanceReport:
    """Result of :func:`energy_balance`.

    Attributes:
        bias_energy_spread: range (max − min) of per-node bias contributions
            ``β·|b_i|``.
        coupling_energy_per_spin: mean total absolute coupling energy per
            variable, ``β · mean_i(Σ_j |J_ij|)``.
        ratio: ``coupling_energy_per_spin / bias_energy_spread``.  Values
            well below 1 mean biases dominate; well above 1 mean couplings
            dominate.
    """

    bias_energy_spread: float
    coupling_energy_per_spin: float
    ratio: float


def energy_balance(
    biases: Shaped[Array, " n"],
    edges: Shaped[Array, "m 2"],
    weights: Shaped[Array, " m"],
    *,
    beta: float = 1.0,
    warn_low: float = 0.05,
    warn_high: float = 2.0,
) -> EnergyBalanceReport:
    r"""Check whether bias and coupling energy scales are balanced.

    Computes the energy contribution from biases vs couplings at the given
    temperature and reports their ratio.  Logs a warning when the ratio
    falls outside ``[warn_low, warn_high]``.

    Args:
        biases: per-node bias array of shape ``(n,)``.
        edges: integer index pairs of shape ``(m, 2)``.
        weights: per-edge coupling of shape ``(m,)``.
        beta: inverse temperature.
        warn_low: ratio below which a warning is logged.
        warn_high: ratio above which a warning is logged.

    Returns:
        An :class:`EnergyBalanceReport`.
    """
    biases = jnp.asarray(biases)
    edges = jnp.asarray(edges)
    weights = jnp.asarray(weights)
    n = biases.shape[0]

    bias_contributions = beta * jnp.abs(biases)
    bias_spread = float(jnp.max(bias_contributions) - jnp.min(bias_contributions))

    # Sum of absolute coupling weights incident on each node.
    abs_w = beta * jnp.abs(weights)
    coupling_per_node = jnp.zeros(n)
    coupling_per_node = coupling_per_node.at[edges[:, 0]].add(abs_w)
    coupling_per_node = coupling_per_node.at[edges[:, 1]].add(abs_w)
    coupling_per_spin = float(jnp.mean(coupling_per_node))

    if bias_spread > 0:
        ratio = coupling_per_spin / bias_spread
    else:
        ratio = float("inf")

    if ratio < warn_low:
        logger.warning(
            "Energy balance ratio %.3f < %.3f: biases dominate, "
            "couplings may be irrelevant.",
            ratio,
            warn_low,
        )
    elif ratio > warn_high:
        logger.warning(
            "Energy balance ratio %.3f > %.1f: couplings dominate, "
            "biases may be irrelevant.",
            ratio,
            warn_high,
        )

    return EnergyBalanceReport(
        bias_energy_spread=bias_spread,
        coupling_energy_per_spin=coupling_per_spin,
        ratio=ratio,
    )

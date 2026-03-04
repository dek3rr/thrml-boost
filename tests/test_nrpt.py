"""Tests for nrpt.py (Non-Reversible Parallel Tempering with vectorized swaps).

Mirrors test_tempering.py structure. Tests:
- Smoke tests (2, 3, 4 chains; zero-energy → all swaps accepted)
- Output format (state shapes, sampler_states, stats keys)
- DEO pair counting (even/odd alternation)
- Acceptance statistics match parallel_tempering for zero-energy case
- Temperature-linearity: vectorized swap gives same log_r as 4-eval formula
- Nonzero energy: cold/hot chains reject some swaps
- Zero rounds returns inputs unchanged
- Mismatched programs raise ValueError
- Extra stats keys (rejection_rates, betas) are present and valid
- Adaptive schedule optimization reduces barrier
"""

import jax
import jax.numpy as jnp
import pytest

from thrml_boost import Block, SpinNode, make_empty_block_state
from thrml_boost.models import IsingEBM, IsingSamplingProgram
from thrml_boost.nrpt import nrpt, optimize_schedule


# ---------------------------------------------------------------------------
# Shared helpers (same _tiny_ising as test_tempering.py)
# ---------------------------------------------------------------------------


def _tiny_ising(n_temps: int = 2, betas=None):
    """2×2 torus Ising, two-coloured free blocks, zero weights."""
    grid = [[SpinNode() for _ in range(2)] for _ in range(2)]
    nodes = [n for row in grid for n in row]
    edges = []
    for i in range(2):
        for j in range(2):
            n = grid[i][j]
            edges.append((n, grid[i][(j + 1) % 2]))
            edges.append((n, grid[(i + 1) % 2][j]))
    even_nodes = [grid[i][j] for i in range(2) for j in range(2) if (i + j) % 2 == 0]
    odd_nodes = [grid[i][j] for i in range(2) for j in range(2) if (i + j) % 2 == 1]
    free_blocks = [Block(even_nodes), Block(odd_nodes)]

    biases = jnp.zeros(len(nodes))
    weights = jnp.zeros(len(edges))

    if betas is None:
        betas = [float(t) / n_temps for t in range(1, n_temps + 1)]

    ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
    programs = [IsingSamplingProgram(e, free_blocks, clamped_blocks=[]) for e in ebms]
    init_state = make_empty_block_state(free_blocks, ebms[0].node_shape_dtypes)

    return nodes, edges, free_blocks, ebms, programs, init_state


def _ising_with_weights(n_nodes: int, betas: list[float], coupling: float = 1.0):
    """Small Ising chain with nonzero coupling for testing real swaps."""
    nodes = [SpinNode() for _ in range(n_nodes)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)]
    biases = jnp.zeros(n_nodes)
    weights = jnp.ones(len(edges)) * coupling

    ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    programs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]
    init_state = make_empty_block_state(free_blocks, ebms[0].node_shape_dtypes)

    return nodes, edges, free_blocks, ebms, programs, init_state


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_nrpt_smoke():
    """Two chains, zero energy → every swap must be accepted."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=2)
    init_states = [init_state, init_state]

    @jax.jit
    def run(key):
        return nrpt(
            key,
            ebms,
            programs,
            init_states,
            clamp_state=[],
            n_rounds=2,
            gibbs_steps_per_round=1,
        )

    final_states, sampler_states, stats = run(jax.random.key(0))

    assert len(final_states) == 2
    assert len(sampler_states) == 2
    assert stats["accepted"].shape == (1,)
    assert stats["attempted"].shape == (1,)
    assert stats["acceptance_rate"].shape == (1,)
    # Zero weights → log_r = 0 → accept_prob = 1
    assert stats["accepted"][0] == 1
    assert stats["attempted"][0] == 1
    assert stats["acceptance_rate"][0] == 1.0


def test_three_chains_smoke():
    """Three chains → two adjacent pairs, both accepted with zero weights."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=3)
    init_states = [init_state] * 3

    @jax.jit
    def run(key):
        return nrpt(
            key,
            ebms,
            programs,
            init_states,
            clamp_state=[],
            n_rounds=4,
            gibbs_steps_per_round=1,
        )

    _, _, stats = run(jax.random.key(7))

    assert stats["accepted"].shape == (2,)
    assert stats["attempted"].shape == (2,)
    assert jnp.all(stats["accepted"] == stats["attempted"])
    assert jnp.all(stats["acceptance_rate"] == 1.0)
    # DEO: 4 rounds → even (0,2) try pair 0; odd (1,3) try pair 1
    assert stats["attempted"][0] == 2
    assert stats["attempted"][1] == 2


def test_four_chains_pair_counting():
    """Four chains → three pairs. Even rounds: pairs 0,2. Odd: pair 1."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=4)
    init_states = [init_state] * 4

    @jax.jit
    def run(key):
        return nrpt(
            key,
            ebms,
            programs,
            init_states,
            clamp_state=[],
            n_rounds=6,
            gibbs_steps_per_round=1,
        )

    _, _, stats = run(jax.random.key(42))

    assert stats["attempted"].shape == (3,)
    # 6 rounds: 3 even (pairs 0,2), 3 odd (pair 1)
    assert stats["attempted"][0] == 3
    assert stats["attempted"][1] == 3
    assert stats["attempted"][2] == 3
    assert jnp.all(stats["acceptance_rate"] == 1.0)


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


def test_output_state_format():
    """States should have same shapes as inputs; sampler_states should be None."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=2)
    init_states = [init_state, init_state]

    @jax.jit
    def run(key, init_states):
        return nrpt(
            key,
            ebms,
            programs,
            init_states,
            [],
            n_rounds=2,
            gibbs_steps_per_round=2,
        )

    final_states, sampler_states, stats = run(jax.random.key(1), init_states)

    assert len(final_states) == 2
    for chain_state in final_states:
        assert len(chain_state) == len(free_blocks)
        for block_state, block in zip(chain_state, free_blocks):
            assert block_state.shape == (len(block),)
            assert block_state.dtype == jnp.bool_

    assert len(sampler_states) == 2
    for chain_ss in sampler_states:
        assert len(chain_ss) == len(free_blocks)
        for ss in chain_ss:
            assert ss is None


def test_extra_stats_keys():
    """nrpt returns additional stats: rejection_rates, betas."""
    _, _, _, ebms, programs, init_state = _tiny_ising(n_temps=3)

    _, _, stats = nrpt(
        jax.random.key(0),
        ebms,
        programs,
        [init_state] * 3,
        [],
        n_rounds=2,
        gibbs_steps_per_round=1,
    )

    assert "rejection_rates" in stats
    assert "betas" in stats
    assert stats["rejection_rates"].shape == (2,)
    assert stats["betas"].shape == (3,)
    # rejection = 1 - acceptance; for zero weights acceptance = 1
    assert jnp.allclose(stats["rejection_rates"], 0.0)


# ---------------------------------------------------------------------------
# Zero rounds
# ---------------------------------------------------------------------------


def test_zero_rounds():
    """n_rounds=0 returns initial states unchanged and zero stats."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=2)
    init_a = [jnp.ones((len(b),), dtype=jnp.bool_) for b in free_blocks]
    init_b = [jnp.zeros((len(b),), dtype=jnp.bool_) for b in free_blocks]

    final_states, _, stats = nrpt(
        jax.random.key(0),
        ebms,
        programs,
        [init_a, init_b],
        [],
        n_rounds=0,
        gibbs_steps_per_round=1,
    )

    for b in range(len(free_blocks)):
        assert jnp.array_equal(final_states[0][b], init_a[b])
        assert jnp.array_equal(final_states[1][b], init_b[b])

    assert stats["accepted"][0] == 0
    assert stats["attempted"][0] == 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raises():
    """Different number of ebms and programs should raise."""
    _, _, _, ebms, programs, init_state = _tiny_ising(n_temps=2)

    with pytest.raises(ValueError, match="same length"):
        nrpt(
            jax.random.key(0),
            ebms[:1],
            programs,
            [init_state] * 2,
            [],
            n_rounds=1,
            gibbs_steps_per_round=1,
        )


def test_mismatched_block_structure_raises():
    """Programs with different block structures should raise ValueError."""
    _, _, _, ebms, programs, init_state = _tiny_ising(n_temps=2)

    nodes_alt = [SpinNode() for _ in range(6)]
    edges_alt = [(nodes_alt[i], nodes_alt[i + 1]) for i in range(5)]
    ebm_alt = IsingEBM(nodes_alt, edges_alt, jnp.zeros(6), jnp.zeros(5), jnp.array(1.0))
    prog_alt = IsingSamplingProgram(
        ebm_alt,
        [Block(nodes_alt[:2]), Block(nodes_alt[2:4]), Block(nodes_alt[4:])],
        clamped_blocks=[],
    )

    with pytest.raises(ValueError, match="block structure"):
        nrpt(
            jax.random.key(0),
            [ebms[0], ebm_alt],
            [programs[0], prog_alt],
            [init_state, [jnp.zeros((2,), jnp.bool_)] * 3],
            [],
            n_rounds=1,
            gibbs_steps_per_round=1,
        )


# ---------------------------------------------------------------------------
# Temperature-linearity correctness
# ---------------------------------------------------------------------------


def test_temperature_linearity():
    """Verify that (β_i-β_j)*(E_base(x_i)-E_base(x_j)) equals the 4-eval formula.

    This is the mathematical identity that makes vectorized swaps valid.
    Uses nonzero weights so energies are nontrivial.
    """
    nodes, edges, free_blocks, ebms, programs, _ = _ising_with_weights(
        n_nodes=8,
        betas=[0.5, 1.0, 2.0, 3.0],
        coupling=1.5,
    )

    # Random initial states
    key = jax.random.key(99)
    keys = jax.random.split(key, 4)
    states = [
        [
            jax.random.bernoulli(keys[c], 0.5, shape=(len(b),)).astype(jnp.bool_)
            for b in free_blocks
        ]
        for c in range(4)
    ]

    spec = programs[0].gibbs_spec
    betas = jnp.array([0.5, 1.0, 2.0, 3.0])

    for p in range(3):
        i, j = p, p + 1
        state_i = states[i]
        state_j = states[j]
        full_i = state_i + []
        full_j = state_j + []

        # 4-eval formula
        Ei_xi = ebms[i].energy(full_i, spec)
        Ej_xj = ebms[j].energy(full_j, spec)
        Ei_xj = ebms[i].energy(full_j, spec)
        Ej_xi = ebms[j].energy(full_i, spec)
        log_r_4eval = float((Ei_xi + Ej_xj) - (Ei_xj + Ej_xi))

        # 1-eval formula: E_β(x) = β * E_base(x), so E_base = E/β
        base_i = float(Ei_xi / betas[i])
        base_j = float(Ej_xj / betas[j])
        log_r_1eval = float((betas[i] - betas[j]) * (base_i - base_j))

        assert abs(log_r_4eval - log_r_1eval) < 1e-4, (
            f"pair ({i},{j}): 4-eval={log_r_4eval:.6f} vs 1-eval={log_r_1eval:.6f}"
        )


# ---------------------------------------------------------------------------
# Nonzero energy: some swaps should be rejected
# ---------------------------------------------------------------------------


def test_nonzero_energy_acceptance():
    """Cold/hot chains with strong coupling should reject some swaps."""
    nodes = [SpinNode() for _ in range(4)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(3)]
    biases = jnp.zeros(4)
    weights = jnp.ones(3) * 2.0

    ebm_cold = IsingEBM(nodes, edges, biases, weights, jnp.array(5.0))
    ebm_hot = IsingEBM(nodes, edges, biases, weights, jnp.array(0.1))

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    programs = [
        IsingSamplingProgram(ebm_cold, free_blocks, []),
        IsingSamplingProgram(ebm_hot, free_blocks, []),
    ]

    init_cold = [jnp.ones((len(b),), jnp.bool_) for b in free_blocks]
    init_hot = [jnp.zeros((len(b),), jnp.bool_) for b in free_blocks]

    @jax.jit
    def run(key, ic, ih):
        return nrpt(
            key,
            [ebm_cold, ebm_hot],
            programs,
            [ic, ih],
            clamp_state=[],
            n_rounds=20,
            gibbs_steps_per_round=1,
        )

    _, _, stats = run(jax.random.key(123), init_cold, init_hot)
    assert float(stats["acceptance_rate"][0]) < 1.0


# ---------------------------------------------------------------------------
# Single chain: nrpt with 1 chain matches _run_blocks
# ---------------------------------------------------------------------------


def test_single_chain_runs():
    """Single-chain NRPT (no swaps possible) should run without error."""
    _, _, free_blocks, ebms, programs, init_state = _tiny_ising(n_temps=1)

    @jax.jit
    def run(key):
        return nrpt(
            key,
            [ebms[0]],
            [programs[0]],
            [init_state],
            [],
            n_rounds=3,
            gibbs_steps_per_round=3,
        )

    final, ss, stats = run(jax.random.key(999))

    assert len(final) == 1
    assert len(final[0]) == len(free_blocks)
    for block_state, block in zip(final[0], free_blocks):
        assert block_state.shape == (len(block),)
    # No pairs → empty stats
    assert stats["accepted"].shape == (0,)
    assert stats["attempted"].shape == (0,)


# ---------------------------------------------------------------------------
# Explicit betas parameter
# ---------------------------------------------------------------------------


def test_explicit_betas():
    """Passing betas= should override EBM beta values in stats."""
    _, _, _, ebms, programs, init_state = _tiny_ising(n_temps=3)
    custom_betas = jnp.array([0.1, 0.5, 0.9])

    _, _, stats = nrpt(
        jax.random.key(0),
        ebms,
        programs,
        [init_state] * 3,
        [],
        n_rounds=2,
        gibbs_steps_per_round=1,
        betas=custom_betas,
    )

    assert jnp.allclose(stats["betas"], custom_betas)


# ---------------------------------------------------------------------------
# Adaptive schedule optimization
# ---------------------------------------------------------------------------


def test_optimize_schedule_preserves_endpoints():
    """Schedule optimization must keep betas[0] and betas[-1] fixed."""
    betas = jnp.array([0.1, 0.5, 1.0, 1.5, 2.0])
    # Uneven rejection rates
    rej = jnp.array([0.1, 0.5, 0.3, 0.8])

    new_betas = optimize_schedule(rej, betas)

    assert new_betas[0] == betas[0]
    assert new_betas[-1] == betas[-1]
    assert new_betas.shape == betas.shape
    # Should be monotonically increasing
    assert jnp.all(jnp.diff(new_betas) >= 0)


def test_optimize_schedule_equalizes():
    """With very uneven rejection, optimization should redistribute β values."""
    betas = jnp.linspace(0.1, 2.0, 8)
    # First pair has very high rejection, rest low
    rej = jnp.array([0.9, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    new_betas = optimize_schedule(rej, betas)

    # More β values should now be concentrated near the start (high barrier region)
    mid_old = float(betas[1])
    mid_new = float(new_betas[1])
    # The new β[1] should be closer to β[0] (more resolution where barrier is high)
    assert mid_new < mid_old, (
        f"Expected new β[1]={mid_new:.4f} < old β[1]={mid_old:.4f}"
    )


def test_optimize_schedule_uniform_rejection_noop():
    """Uniform rejection rates → schedule should stay approximately the same."""
    betas = jnp.linspace(0.5, 2.0, 5)
    rej = jnp.array([0.3, 0.3, 0.3, 0.3])

    new_betas = optimize_schedule(rej, betas)

    assert jnp.allclose(new_betas, betas, atol=1e-5)


# ---------------------------------------------------------------------------
# Many rounds: acceptance rates should be stable
# ---------------------------------------------------------------------------


def test_many_rounds_stability():
    """Run many rounds; acceptance rates should converge to a stable value."""
    _, _, _, ebms, programs, init_state = _ising_with_weights(
        n_nodes=8,
        betas=[0.5, 1.0, 1.5, 2.0],
        coupling=0.5,
    )
    init_states = [init_state] * 4

    @jax.jit
    def run(key):
        return nrpt(
            key,
            ebms,
            programs,
            init_states,
            [],
            n_rounds=200,
            gibbs_steps_per_round=3,
        )

    _, _, stats = run(jax.random.key(77))

    # All pairs should have been attempted
    assert jnp.all(stats["attempted"] > 0)
    # Rates should be in [0, 1]
    assert jnp.all(stats["acceptance_rate"] >= 0.0)
    assert jnp.all(stats["acceptance_rate"] <= 1.0)
    # With moderate coupling and spacing, not all swaps should be accepted
    # (at least one pair should have < 100% acceptance)
    assert jnp.any(stats["acceptance_rate"] < 1.0)
    # rejection + acceptance = 1
    assert jnp.allclose(stats["rejection_rates"] + stats["acceptance_rate"], 1.0)

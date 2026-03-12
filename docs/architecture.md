# Architecture

Hamon's internals are organized around one idea: **block-structured Gibbs
sampling on padded, stacked PyTree states, compiled once via JAX**.

## The core abstraction

A `BlockSamplingProgram` holds everything needed to run block Gibbs on a single
model: the `BlockGibbsSpec` (which blocks to sample, which to clamp), the
conditional samplers for each block, and the factors that define the energy.

The `BlockSpec` manages the mapping between block-local and global state arrays.
Because JAX requires rectangular arrays, variable-size blocks are padded and
stacked by structural group. This costs some wasted FLOPs but avoids Python-level
loops over blocks, which would cause XLA compile times to scale linearly with
block count.

## State representation

Global state is a list of arrays, one per structural group of blocks. Each array
has shape `(n_blocks_in_group, max_block_size, ...)` with padding where needed.
`BlockSpec` tracks the valid (non-padded) indices so that `block_state_to_global`
and `from_global_state` can convert between representations without data loss.

Node identity matters: the same `SpinNode()` object must appear in both the EBM
and the block definitions. Hamon enforces this through `init_factory`, which
always extracts `free_blocks = programs[0].gibbs_spec.free_blocks` rather than
capturing block objects from an outer scope.

## Parallel tempering layout

NRPT runs \( N \) chains, each at a different inverse temperature \( \beta_i \).
Rather than looping over chains in Python (which unrolls \( N \) copies of the
computation graph in XLA), Hamon stacks all chain states into a leading dimension
and uses `jax.vmap` for the Gibbs sweeps.

The tempering round is a `lax.scan` loop:

```
for round in range(n_rounds):
    # 1. Gibbs sweeps (vmapped across chains)
    states = vmap(gibbs_sweep)(states)

    # 2. DEO swap proposals (single parity)
    parity = round % 2
    states, accepted = deo_swap(states, energies, parity)
```

Swap decisions use the Metropolis-Hastings criterion with the
temperature-linearity trick: for Ising models, \( E(\beta, x) = \beta \cdot E(1, x) \),
so swap acceptance ratios reduce to
\( \exp\bigl((\beta_{i+1} - \beta_i)(V^{(i+1)} - V^{(i)})\bigr) \)
without recomputing energies at each temperature.

## Energy caching

When an `energy_delta_fn` is provided (e.g. from `make_ising_delta_fn`), Hamon
computes the full energy only once at round 0, then maintains a running cache
that is updated incrementally after each Gibbs sweep. After swaps, the cache is
permuted to match the new chain ordering.

This eliminates the \( O(|\mathcal{E}|) \) energy recomputation that would
otherwise dominate each round.

## Index process tracking

The index process tracks which chain index occupies which temperature level over
time. Hamon uses a permutation-based representation: `index_state` is an array of
shape `(n_chains, 2)` where column 0 is the current position in the ladder and
column 1 is the direction of travel.

Because DEO swaps are disjoint transpositions, the swap permutation is
self-inverse (`inv_perm == perm`), so updating the index state after a swap
requires only a single gather — no `jnp.argsort`.

Round-trip counting and \( \Lambda \) estimation happen in `round_trip_summary`
from the accumulated index state and rejection rates.

## Schedule optimization

`optimize_schedule` implements the equi-acceptance reparameterization from
Algorithm 4 of Syed et al. (2021). Given observed rejection rates
\( r_0, \ldots, r_{N-2} \), it constructs the CDF of the rejection cost and
inverts it to find \( \beta \) values that equalize the per-level contribution
to \( \Lambda \).

`nrpt_adaptive` wraps this in a loop: run a short burn-in, measure rejections,
reposition betas, repeat for `n_tune` phases, then run production.

## Dynamic blocks

For models where different temperature levels benefit from different block
granularity, `dynamic_blocks` provides influence-aware partitioning.
`compute_aggregate_influence` measures how strongly each node couples to its
neighbors. `influence_aware_partition` then groups nodes into blocks of a
target size, keeping strongly-coupled nodes together. `per_temperature_block_config`
can assign different block sizes to different chains based on temperature.

## Class hierarchy

### Factors

```
AbstractFactor
├── WeightedFactor
│   └── DiscreteEBMFactor
│       ├── SquareDiscreteEBMFactor
│       │   ├── SpinEBMFactor
│       │   └── SquareCategoricalEBMFactor
│       └── CategoricalEBMFactor
└── EBMFactor
```

### Conditional samplers

```
AbstractConditionalSampler
└── AbstractParametricConditionalSampler
    ├── BernoulliConditional
    │   └── SpinGibbsConditional
    └── SoftmaxConditional
        └── CategoricalGibbsConditional
```

### Observers

```
AbstractObserver
├── StateObserver
└── MomentAccumulatorObserver
```

### Models

```
AbstractEBM
└── AbstractFactorizedEBM
    └── IsingEBM
```

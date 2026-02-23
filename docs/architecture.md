# Architecture

## What is THRML?

THRML is a [JAX](https://docs.jax.dev/en/latest/)-based Python library for efficient [block Gibbs sampling](https://proceedings.mlr.press/v15/gonzalez11a/gonzalez11a.pdf) of graphical models at scale. It provides the machinery for blocked Gibbs sampling on any graphical model and includes built-in support for [Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/cogscibm.pdf) and other discrete energy-based models.

THRML was originally developed by [Extropic AI](https://github.com/Extropic-AI/thrml). THRML-Boost is a performance-optimized fork that preserves the full API while improving JAX compilation and runtime efficiency.

## Core concepts

From a user perspective there are three main components: **blocks**, **factors**, and **programs**. For worked examples see the example notebooks.

### Blocks

A `Block` is a collection of nodes of the same type with implicit ordering. Blocks are the unit of parallelism in block Gibbs sampling — all nodes in a block are updated simultaneously in a single SIMD-friendly JAX operation.

### Factors

Factors organize interactions between variables into a [bipartite factor graph](https://ocw.mit.edu/courses/6-438-algorithms-for-inference-fall-2014/3e3e9934d12e3537b4e9b46b53cd5bf1_MIT6_438F14_Lec4.pdf). Each factor synthesizes a batch of `InteractionGroup`s and must implement `to_interaction_groups()`. An `InteractionGroup` specifies directed computational dependencies: which nodes to update (head), which neighbor states to read (tail), and what static parameters (weights) to use.

### Programs

Programs are the orchestrating data structures. `BlockSamplingProgram` handles the mapping and bookkeeping for padded block Gibbs sampling — managing global state representations efficiently for JAX. `FactorSamplingProgram` is a convenience wrapper that converts factors into interaction groups automatically. Programs coordinate free/clamped blocks, conditional samplers, and interactions to execute the sampling loop.

## Internal design

The core approach is to represent everything as contiguous arrays and PyTrees, operate on these flat structures, and map to/from them at the user boundary. Internally this is the "global state" (as opposed to the user-facing "block state"). This is similar in spirit to struct-of-arrays (SoA) layout and to other JAX graphical model packages like [PGMax](https://github.com/google-deepmind/PGMax).

An important distinction from PGMax is that THRML supports **PyTree states and heterogeneous node types**. Heterogeneity is handled by splitting nodes according to their PyTree structure and organizing the global state as a list of these PyTrees, stacked across blocks that share the same structure. The management of these indices and the mapping between block and global representations is constructed and held by the program's `BlockSpec`.

Since JAX does not support ragged arrays, every block within a structural group must have the same leaf array size. THRML handles variable block sizes by stacking and padding. There is an inherent tradeoff: padding wastes some compute, but the alternative — looping over blocks in Python — incurs untenable XLA compile-time cost.

Everything else exists to provide convenience for creating and working with a program. With a tight core focused on block index management and padding, the codebase stays lightweight and hackable.

## What THRML-Boost optimizes

THRML-Boost does not change the mathematical semantics of any sampler. It targets the JAX compilation and runtime overhead:

- **`jax.vmap` parallel tempering** — replaces the Python for-loop over chains that unrolled N copies of the full Gibbs graph into XLA. One kernel, all chains, constant compile time.
- **`jax.lax.scan` carry threading** — global state is now carried through the scan loop instead of being rebuilt from block states on every iteration.
- **Fixed accumulator dtype** — `MomentAccumulatorObserver` no longer infers dtype per step; set once at construction (float32 default).
- **Pre-built `BlockSpec` passthrough** — `energy()` accepts a pre-built spec to avoid reconstructing it on every call (critical during tempering swap attempts).
- **Deterministic global state ordering** — `dict.fromkeys()` replaces `set()` for deduplication, so ordering is reproducible across runs.
- **Ragged `hinton_init`** — correctly handles blocks of different sizes.


## Class hierarchy

### Factors

```
AbstractFactor
├── WeightedFactor
│   └── DiscreteEBMFactor — spin × categorical polynomial interactions
│       ├── SquareDiscreteEBMFactor — merged groups for square weight tensors
│       │   ├── SpinEBMFactor — spin-only
│       │   └── SquareCategoricalEBMFactor — categorical-only (square)
│       └── CategoricalEBMFactor — categorical-only (general)
└── EBMFactor — factors with energy functions
```

### Conditional samplers

```
AbstractConditionalSampler
└── AbstractParametricConditionalSampler
    ├── BernoulliConditional — spin-valued Bernoulli sampling
    │   └── SpinGibbsConditional — Gibbs updates for spin EBMs
    └── SoftmaxConditional — categorical softmax sampling
        └── CategoricalGibbsConditional — Gibbs updates for categorical EBMs
```

### Observers

```
AbstractObserver
├── StateObserver — records raw states at each observation step
└── MomentAccumulatorObserver — online moment accumulation (mean, variance, etc.)
```

### EBMs

```
AbstractEBM
└── AbstractFactorizedEBM — energy = sum of factor energies
    └── IsingEBM — standard Ising model (biases + pairwise couplings)
```

### Programs

```
BlockSamplingProgram — core padded block Gibbs engine
└── FactorSamplingProgram — auto-converts factors → interaction groups
    └── IsingSamplingProgram — thin Ising-specific wrapper
```

## Limitations

Sampling is a fundamentally hard problem. Generating samples from a high-dimensional distribution can require many steps even with parallelized proposals. THRML is focused on Gibbs sampling; for general sampling it is not always clear when Gibbs is substantially [faster](https://arxiv.org/abs/2007.08200) or [slower](https://arxiv.org/abs/1605.00139) than other MCMC methods. As a concrete example: a two-node Ising model with $J = -\infty, h = 0$ has two ground states $\{-1,-1\}$ and $\{1,1\}$; Gibbs sampling will never mix between them, while uniform Metropolis–Hastings converges quickly.

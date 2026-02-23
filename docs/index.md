<div align="center">
  <img src="_static/logo/logo.svg" alt="THRML-Boost Logo" width="200" style="margin-bottom: 20px;">
</div>

# **THRML-Boost**

*Performance-optimized block Gibbs sampling for probabilistic graphical models in JAX.*

---

THRML-Boost is a fork of [Extropic AI's THRML library](https://github.com/Extropic-AI/thrml) that targets the JAX compilation and runtime bottlenecks in the original implementation. The API is unchanged — it's a drop-in replacement.

The library provides GPU-accelerated tools for blocked Gibbs sampling on sparse, heterogeneous graphs. It's a good fit for Ising models, Boltzmann machines, discrete energy-based models, or anything with a bipartite factor-graph structure.

**What's different from upstream THRML:**

- Parallel tempering chains run via `jax.vmap` instead of a Python loop — constant compile time, better GPU utilization
- Global state threaded through `jax.lax.scan` carry — no redundant rebuilds each iteration
- Moment accumulator dtype fixed at construction — avoids silent float64 promotion on GPU
- `BlockSpec` pre-built and reused in energy evaluation — eliminates per-call reconstruction
- Deterministic global state ordering — reproducible across runs

See the [architecture guide](architecture.md) for how the internals work, or jump straight to the [API reference](api/block_management.md).

## Installation

Requires Python ≥ 3.10 and a working [JAX installation](https://jax.readthedocs.io/en/latest/installation.html).

```bash
git clone https://github.com/dek3rr/thrml-boost.git
cd thrml-boost
pip install -e .
```

For notebooks and examples:

```bash
pip install -e ".[examples]"
```

## Quick example

Sample a small Ising chain with two-color block Gibbs:

```python
import jax
import jax.numpy as jnp
from thrml_boost import SpinNode, Block, SamplingSchedule, sample_states
from thrml_boost.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
biases = jnp.zeros((5,))
weights = jnp.ones((4,)) * 0.5
beta = jnp.array(1.0)
model = IsingEBM(nodes, edges, biases, weights, beta)

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

key = jax.random.key(0)
k_init, k_samp = jax.random.split(key, 2)
init_state = hinton_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
```

## Attribution

THRML-Boost is a derivative work of [thrml](https://github.com/Extropic-AI/thrml) by Extropic AI, licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See the [NOTICE](https://github.com/dek3rr/thrml-boost/blob/main/NOTICE) file for details.

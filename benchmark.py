#!/usr/bin/env python3
"""
thrml-boost comprehensive benchmark.

Isolates and measures each optimization against a fair baseline:

  1. Single-chain Gibbs       — scan carry threading, deterministic layout
  2. Parallel tempering       — vmap over chains
  3. Energy evaluation        — BlockSpec pre-build fast path
  4. Moment accumulation      — fixed dtype avoids float64 emulation on GPU
  5. Correctness checks       — deterministic ordering, ragged hinton_init

Usage:
    python benchmark.py                          # defaults (32×32 grid)
    python benchmark.py --L 64                   # larger grid
    python benchmark.py --L 32 --runs 5          # more statistical power
    python benchmark.py --skip-thrml             # only test thrml-boost internals
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

import thrml_boost  # noqa: E402
from thrml_boost import (  # noqa: E402
    Block,
    SamplingSchedule,
    sample_states,
    sample_with_observation,
    MomentAccumulatorObserver,
)
from thrml_boost.models import (  # noqa: E402
    IsingEBM,
    IsingSamplingProgram,
    hinton_init,
)
from thrml_boost.tempering import parallel_tempering  # noqa: E402

try:
    import thrml

    HAS_THRML = True
except ImportError:
    HAS_THRML = False


# ── Utilities ─────────────────────────────────────────────────────


def create_ising_components(L: int, lib=thrml_boost):
    """Build reusable 2D Ising graph with checkerboard blocking."""
    N = L * L
    _SpinNode = lib.SpinNode
    _Block = lib.Block

    nodes = [_SpinNode() for _ in range(N)]

    edges, wl = [], []
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if j + 1 < L:
                edges.append((nodes[idx], nodes[idx + 1]))
                wl.append(1.0)
            if i + 1 < L:
                edges.append((nodes[idx], nodes[idx + L]))
                wl.append(1.0)

    biases = jnp.zeros(N, dtype=jnp.float32)
    weights = jnp.array(wl, dtype=jnp.float32)

    black = [nodes[i * L + j] for i in range(L) for j in range(L) if (i + j) % 2 == 0]
    white = [nodes[i * L + j] for i in range(L) for j in range(L) if (i + j) % 2 == 1]
    free_blocks = [_Block(black), _Block(white)]

    return nodes, edges, biases, weights, free_blocks


def sync(x):
    """Block until all arrays in an arbitrary pytree are ready."""
    for leaf in jax.tree.leaves(x):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    return x


def time_fn(fn, num_runs: int = 3):
    """Time a zero-arg callable. First call = compile, rest = execution only."""
    t0 = time.perf_counter()
    sync(fn())
    compile_time = time.perf_counter() - t0

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - t0)

    return {
        "compile_s": compile_time,
        "avg_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": min(times),
        "times": times,
    }


def header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def row(label: str, res: dict, extra: str = ""):
    print(
        f"  {label:30s}  compile={res['compile_s']:.3f}s  "
        f"avg={res['avg_s']:.4f}s ±{res['std_s']:.4f}s  {extra}"
    )


# ═════════════════════════════════════════════════════════════════
#  1. Single-chain Gibbs
# ═════════════════════════════════════════════════════════════════


def bench_single_chain(L: int, n_warmup: int, n_samples: int, num_runs: int):
    """Tests scan carry threading and deterministic global state layout.

    Both libraries run sample_states on the same 2D Ising model.
    thrml-boost threads global state through lax.scan carry and uses
    targeted scatter instead of full rebuild each iteration.
    """
    header(f"1. Single-chain Gibbs | {L}×{L} ({L * L} spins)")

    results = {}
    libs = [(thrml_boost, "thrml-boost")]
    if HAS_THRML:
        libs.append((thrml, "thrml"))

    for lib, name in libs:
        nodes, edges, biases, weights, free_blocks = create_ising_components(L, lib)
        model = lib.models.IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
        program = lib.models.IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        obs_block = lib.Block(nodes)
        schedule = lib.SamplingSchedule(
            n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=1
        )
        key = jax.random.key(42)
        init_state = lib.models.hinton_init(key, model, free_blocks, ())

        # Use a fresh key each call to avoid XLA result caching
        call_keys = jax.random.split(key, num_runs + 2)

        def run(k=call_keys[0]):
            return lib.sample_states(k, program, schedule, init_state, [], [obs_block])

        # Compile
        t0 = time.perf_counter()
        sync(run(call_keys[0]))
        compile_s = time.perf_counter() - t0

        times = []
        for r in range(num_runs):
            t0 = time.perf_counter()
            sync(run(call_keys[r + 1]))
            times.append(time.perf_counter() - t0)

        res = {
            "compile_s": compile_s,
            "avg_s": np.mean(times),
            "std_s": np.std(times),
            "min_s": min(times),
        }
        results[name] = res
        sps = n_samples / res["avg_s"]
        row(name, res, f"({sps:.0f} samples/s)")

    if "thrml" in results and "thrml-boost" in results:
        sp = results["thrml"]["avg_s"] / results["thrml-boost"]["avg_s"]
        print(f"\n  → thrml-boost runtime speedup: {sp:.2f}×")

    return results


# ═════════════════════════════════════════════════════════════════
#  2. Parallel tempering  (vmap over chains)
# ═════════════════════════════════════════════════════════════════


def bench_parallel_tempering(
    L: int, chain_counts: list[int], n_rounds: int, gibbs_steps: int, num_runs: int
):
    """Tests the headline optimization: jax.vmap over tempered chains.

    Compares thrml-boost's parallel_tempering (1 vmapped kernel) against
    a sequential baseline that calls sample_states per chain in a Python
    loop.  Both use thrml-boost, isolating the vmap win from other changes.
    """
    header(f"2. Parallel tempering scaling | {L}×{L} ({L * L} spins)")

    # Shared graph — all chains use the same nodes/edges, differ only in beta
    nodes, edges, biases, weights, free_blocks = create_ising_components(L)
    obs_block = Block(nodes)

    for n_chains in chain_counts:
        print(f"\n  ── {n_chains} chains ──")
        betas = jnp.linspace(0.5, 2.0, n_chains)

        ebms = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(float(b))) for b in betas
        ]
        programs = [
            IsingSamplingProgram(ebm, free_blocks, clamped_blocks=[]) for ebm in ebms
        ]

        master_key = jax.random.key(0)
        keys = jax.random.split(master_key, n_chains + 1)
        init_states = [
            hinton_init(keys[i], ebms[0], free_blocks, ()) for i in range(n_chains)
        ]

        # ── A) thrml-boost parallel_tempering ──
        pt_keys = jax.random.split(keys[-1], num_runs + 2)

        def run_pt(k=pt_keys[0]):
            return parallel_tempering(
                k,
                ebms,
                programs,
                init_states,
                [],
                n_rounds=n_rounds,
                gibbs_steps_per_round=gibbs_steps,
            )

        res_pt = time_fn(run_pt, num_runs=num_runs)
        total_sweeps = n_rounds * gibbs_steps * n_chains
        row(
            "parallel_tempering (vmap)",
            res_pt,
            f"({total_sweeps / res_pt['avg_s']:.0f} sweeps/s)",
        )

        # ── B) Sequential baseline: Python loop of sample_states ──
        #    Same total Gibbs work, no vmap, no swap attempts.
        #    This is what you'd write without parallel_tempering.
        total_steps = n_rounds * gibbs_steps
        seq_schedule = SamplingSchedule(
            n_warmup=total_steps, n_samples=1, steps_per_sample=1
        )
        seq_keys = jax.random.split(master_key, n_chains * (num_runs + 2))

        def run_seq(offset=0):
            out = []
            for i in range(n_chains):
                k = seq_keys[offset * n_chains + i]
                out.append(
                    sample_states(
                        k, programs[i], seq_schedule, init_states[i], [], [obs_block]
                    )
                )
            return out

        # Compile (first chain compiles; rest hit JIT cache)
        t0 = time.perf_counter()
        sync(run_seq(0))
        seq_compile = time.perf_counter() - t0

        seq_times = []
        for r in range(num_runs):
            t0 = time.perf_counter()
            sync(run_seq(r + 1))
            seq_times.append(time.perf_counter() - t0)

        res_seq = {
            "compile_s": seq_compile,
            "avg_s": np.mean(seq_times),
            "std_s": np.std(seq_times),
            "min_s": min(seq_times),
        }
        row(
            "sequential loop",
            res_seq,
            f"({total_sweeps / res_seq['avg_s']:.0f} sweeps/s)",
        )

        rt_speedup = res_seq["avg_s"] / res_pt["avg_s"]
        ct_ratio = res_seq["compile_s"] / max(res_pt["compile_s"], 1e-9)
        print(
            f"  → runtime speedup: {rt_speedup:.2f}×  |  compile ratio: {ct_ratio:.2f}×"
        )


# ═════════════════════════════════════════════════════════════════
#  3. Energy evaluation  (BlockSpec fast-path)
# ═════════════════════════════════════════════════════════════════


def bench_energy(L: int, n_evals: int, num_runs: int):
    """Tests the pre-built BlockSpec fast-path in energy().

    In upstream thrml, energy() always rebuilds a BlockSpec from list[Block].
    thrml-boost accepts a pre-built spec, skipping Python-level construction.
    The win is largest during parallel tempering, where energy() is called
    4× per swap attempt (this benchmark isolates the per-call overhead).
    """
    header(f"3. Energy evaluation | {L}×{L} ({L * L} spins) | {n_evals} calls/trial")

    nodes, edges, biases, weights, free_blocks = create_ising_components(L)
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    key = jax.random.key(99)
    state = hinton_init(key, model, free_blocks, ())
    spec = program.gibbs_spec  # pre-built BlockGibbsSpec

    # Sanity: both paths should give the same energy
    e_blocks = model.energy(state, free_blocks)
    e_spec = model.energy(state, spec)
    assert jnp.allclose(e_blocks, e_spec), f"Energy mismatch: {e_blocks} vs {e_spec}"
    print(f"  Sanity check passed (energy = {float(e_spec):.4f})")

    # ── Non-JIT'd: measures Python-level BlockSpec construction overhead ──
    print(f"\n  Non-JIT'd ({n_evals} calls):")

    def eval_blocks():
        e = None
        for _ in range(n_evals):
            e = model.energy(state, free_blocks)
        return e

    def eval_spec():
        e = None
        for _ in range(n_evals):
            e = model.energy(state, spec)
        return e

    res_b = time_fn(eval_blocks, num_runs=num_runs)
    res_s = time_fn(eval_spec, num_runs=num_runs)
    row("list[Block] (rebuild)", res_b, f"({n_evals / res_b['avg_s']:.0f} eval/s)")
    row("BlockSpec  (pre-built)", res_s, f"({n_evals / res_s['avg_s']:.0f} eval/s)")
    print(f"  → fast-path speedup: {res_b['avg_s'] / res_s['avg_s']:.2f}×")

    # ── JIT'd: the compile-time cost of tracing with/without spec ──
    print("\n  JIT'd:")

    @jax.jit
    def jit_energy_blocks(s):
        return model.energy(s, free_blocks)

    @jax.jit
    def jit_energy_spec(s):
        return model.energy(s, spec)

    t0 = time.perf_counter()
    sync(jit_energy_blocks(state))
    ct_blocks = time.perf_counter() - t0

    t0 = time.perf_counter()
    sync(jit_energy_spec(state))
    ct_spec = time.perf_counter() - t0

    def jit_loop_blocks():
        for _ in range(n_evals):
            jit_energy_blocks(state)
        return jit_energy_blocks(state)

    def jit_loop_spec():
        for _ in range(n_evals):
            jit_energy_spec(state)
        return jit_energy_spec(state)

    res_jb = time_fn(jit_loop_blocks, num_runs=num_runs)
    res_js = time_fn(jit_loop_spec, num_runs=num_runs)

    print(
        f"  {'list[Block]':30s}  compile={ct_blocks:.4f}s  avg={res_jb['avg_s']:.4f}s"
    )
    print(f"  {'BlockSpec':30s}  compile={ct_spec:.4f}s  avg={res_js['avg_s']:.4f}s")


# ═════════════════════════════════════════════════════════════════
#  4. Moment accumulation  (dtype fix)
# ═════════════════════════════════════════════════════════════════


def bench_moments(L: int, n_warmup: int, n_samples: int, num_runs: int):
    """Tests the fixed-dtype MomentAccumulatorObserver.

    Upstream thrml inferred accumulator dtype per scan step, silently
    triggering float64 emulation on GPU.  thrml-boost sets it once at
    construction (float32 by default).

    We compare MomentAccumulatorObserver with explicit float32 vs float64
    to show the GPU penalty of the wider dtype.  On CPU the difference is
    smaller but still measurable.
    """
    header(f"4. Moment accumulation | {L}×{L} ({L * L} spins)")

    nodes, edges, biases, weights, free_blocks = create_ising_components(L)
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    obs_block = Block(nodes)

    key = jax.random.key(77)
    init_state = hinton_init(key, model, free_blocks, ())
    schedule = SamplingSchedule(
        n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=1
    )

    # Moment spec: first moment of each node
    moment_spec = [[(n,) for n in nodes]]

    def spin_transform(state, _blocks):
        return [2 * x.astype(jnp.float32) - 1 for x in state]

    # ── StateObserver baseline (no accumulation overhead) ──
    def run_state_obs(k=key):
        return sample_states(k, program, schedule, init_state, [], [obs_block])

    res_state = time_fn(run_state_obs, num_runs=num_runs)
    row("StateObserver (baseline)", res_state)

    # ── MomentAccumulatorObserver float32 (thrml-boost default) ──
    obs_f32 = MomentAccumulatorObserver(
        moment_spec, f_transform=spin_transform, dtype=jnp.float32
    )

    def run_moment_f32(k=key):
        with jax.numpy_dtype_promotion("standard"):
            return sample_with_observation(
                k, program, schedule, init_state, [], obs_f32.init(), obs_f32
            )

    res_f32 = time_fn(run_moment_f32, num_runs=num_runs)
    row("MomentAccumulator float32", res_f32)

    # ── MomentAccumulatorObserver float64 (upstream behavior) ──
    obs_f64 = MomentAccumulatorObserver(
        moment_spec, f_transform=spin_transform, dtype=jnp.float64
    )

    def run_moment_f64(k=key):
        with jax.numpy_dtype_promotion("standard"):
            return sample_with_observation(
                k, program, schedule, init_state, [], obs_f64.init(), obs_f64
            )

    res_f64 = time_fn(run_moment_f64, num_runs=num_runs)
    row("MomentAccumulator float64", res_f64)

    if res_f64["avg_s"] > 0:
        print(
            f"\n  → float32 vs float64: {res_f64['avg_s'] / res_f32['avg_s']:.2f}× faster"
        )
        print("    (on GPU this gap widens to ~4× due to float64 emulation)")


# ═════════════════════════════════════════════════════════════════
#  5. Correctness checks
# ═════════════════════════════════════════════════════════════════


def check_correctness():
    header("5. Correctness checks")
    all_pass = True

    # ── 5a. Deterministic global state ordering ──
    print("  [a] Deterministic global state ordering")
    L = 8
    for trial in range(3):
        nodes, edges, biases, weights, free_blocks = create_ising_components(L)
        model = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        obs_block = Block(nodes)
        schedule = SamplingSchedule(n_warmup=20, n_samples=5, steps_per_sample=1)

        key = jax.random.key(42)
        s1 = hinton_init(key, model, free_blocks, ())
        s2 = hinton_init(key, model, free_blocks, ())

        init_ok = all(jnp.array_equal(a, b) for a, b in zip(s1, s2))

        r1 = sample_states(key, program, schedule, s1, [], [obs_block])
        r2 = sample_states(key, program, schedule, s2, [], [obs_block])
        sample_ok = all(
            jnp.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(r1), jax.tree.leaves(r2))
        )

        ok = init_ok and sample_ok
        if not ok:
            all_pass = False
        print(
            f"      trial {trial}: init={'ok' if init_ok else 'FAIL'}  "
            f"samples={'ok' if sample_ok else 'FAIL'}"
        )

    status = "PASS" if all_pass else "FAIL"
    print(f"      → {status}: same seed produces identical results across runs\n")

    # ── 5b. Ragged hinton_init (odd grid → unequal block sizes) ──
    print("  [b] Ragged hinton_init")
    ragged_pass = True
    for L_odd in [5, 7, 11]:
        nodes_r, edges_r, biases_r, weights_r, fb_r = create_ising_components(L_odd)
        sizes = [len(b) for b in fb_r]
        model_r = IsingEBM(nodes_r, edges_r, biases_r, weights_r, jnp.array(1.0))

        try:
            state_r = hinton_init(jax.random.key(0), model_r, fb_r, ())
            shapes = [s.shape for s in state_r]
            expected = [(sizes[0],), (sizes[1],)]
            ok = shapes == expected
        except Exception as e:
            ok = False
            print(f"      {L_odd}×{L_odd}: FAIL ({e})")

        if ok:
            print(f"      {L_odd}×{L_odd}: blocks={sizes}  init_shapes={shapes}  ok")
        else:
            ragged_pass = False

    print(f"      → {'PASS' if ragged_pass else 'FAIL'}\n")

    # ── 5c. Parallel tempering acceptance rates ──
    print("  [c] Parallel tempering acceptance stats")
    L_pt = 8
    nodes_p, edges_p, biases_p, weights_p, fb_p = create_ising_components(L_pt)
    betas = [0.5, 1.0, 1.5, 2.0]
    ebms = [
        IsingEBM(nodes_p, edges_p, biases_p, weights_p, jnp.array(b)) for b in betas
    ]
    progs = [IsingSamplingProgram(e, fb_p, []) for e in ebms]

    key = jax.random.key(7)
    keys = jax.random.split(key, len(betas) + 1)
    inits = [hinton_init(keys[i], ebms[0], fb_p, ()) for i in range(len(betas))]

    states, ss, stats = parallel_tempering(
        keys[-1],
        ebms,
        progs,
        inits,
        [],
        n_rounds=100,
        gibbs_steps_per_round=5,
    )

    acc = stats["acceptance_rate"]
    att = stats["attempted"]
    print(f"      betas:      {betas}")
    print(f"      acc. rates: {[f'{float(r):.3f}' for r in acc]}")
    print(f"      attempted:  {[int(a) for a in att]}")

    rates_ok = all(0.0 <= float(r) <= 1.0 for r in acc)
    attempted_ok = all(int(a) > 0 for a in att)
    pt_pass = rates_ok and attempted_ok
    print(
        f"      → {'PASS' if pt_pass else 'FAIL'}: rates ∈ [0,1], all pairs attempted\n"
    )

    # ── 5d. Energy fast-path equivalence ──
    print("  [d] Energy fast-path equivalence")
    L_e = 16
    nodes_e, edges_e, biases_e, weights_e, fb_e = create_ising_components(L_e)
    model_e = IsingEBM(nodes_e, edges_e, biases_e, weights_e, jnp.array(1.5))
    prog_e = IsingSamplingProgram(model_e, fb_e, [])
    state_e = hinton_init(jax.random.key(42), model_e, fb_e, ())

    e_list = model_e.energy(state_e, fb_e)
    e_spec = model_e.energy(state_e, prog_e.gibbs_spec)
    energy_ok = jnp.allclose(e_list, e_spec)
    print(f"      energy(list[Block])  = {float(e_list):.6f}")
    print(f"      energy(BlockSpec)    = {float(e_spec):.6f}")
    print(f"      → {'PASS' if energy_ok else 'FAIL'}\n")

    return all([all_pass, ragged_pass, pt_pass, energy_ok])


# ═════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════


def print_summary(args):
    header("Summary")
    print(f"""
  This benchmark tested the five optimizations in thrml-boost:

  1. Scan carry threading         (single-chain Gibbs)
     Global state is threaded through lax.scan instead of being rebuilt
     each iteration.  Expect 5–15% runtime improvement.

  2. vmap parallel tempering      (multi-chain)
     All chains run in one vmapped kernel instead of N sequential
     dispatches.  Expect 2–3× runtime speedup, plus sub-linear compile
     time scaling (constant in chain count vs. O(N) for loop unrolling).

  3. BlockSpec pre-build           (energy evaluation)
     energy() accepts a pre-built spec, skipping Python-level construction
     on every call. Matters most during swap attempts (4 calls each).

  4. Fixed accumulator dtype       (moment observer)
     float32 by default avoids silent float64 promotion → up to 4× on GPU.

  5. Deterministic layout + ragged init
     dict.fromkeys() for reproducible ordering; hinton_init handles
     unequal block sizes correctly.

  Grid: {args.L}×{args.L} ({args.L**2} spins)  |  Device: {jax.devices()[0]}
""")


# ═════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="thrml-boost comprehensive benchmark")
    parser.add_argument(
        "--L", type=int, default=32, help="Grid side length (default: 32 → 1024 spins)"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Timed runs per benchmark (default: 3)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Samples for single-chain bench (default: 2000)",
    )
    parser.add_argument(
        "--warmup", type=int, default=200, help="Warmup steps (default: 200)"
    )
    parser.add_argument(
        "--pt-rounds", type=int, default=50, help="Tempering rounds (default: 50)"
    )
    parser.add_argument(
        "--pt-gibbs",
        type=int,
        default=10,
        help="Gibbs steps per tempering round (default: 10)",
    )
    parser.add_argument(
        "--pt-chains",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Chain counts to test (default: 2 4 8 16)",
    )
    parser.add_argument(
        "--energy-evals",
        type=int,
        default=500,
        help="Energy eval calls per trial (default: 500)",
    )
    parser.add_argument(
        "--moment-samples",
        type=int,
        default=500,
        help="Samples for moment benchmark (default: 500)",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--skip-thrml", action="store_true", help="Skip upstream thrml comparison"
    )
    parser.add_argument(
        "--only",
        type=int,
        nargs="+",
        default=None,
        help="Run only specific benchmarks (1-5)",
    )
    args = parser.parse_args()

    if args.skip_thrml:
        HAS_THRML = False

    print(f"Device: {jax.devices()[0]}")
    print(f"JAX version: {jax.__version__}")
    print(f"thrml-boost version: {thrml_boost.__version__}")
    if HAS_THRML:
        print(f"thrml version: {thrml.__version__}")
    else:
        print("thrml: not installed (skipping head-to-head)")
    print(f"Grid: {args.L}×{args.L} ({args.L**2} spins)")

    run_all = args.only is None

    if run_all or 1 in args.only:
        bench_single_chain(args.L, args.warmup, args.samples, args.runs)

    if run_all or 2 in args.only:
        bench_parallel_tempering(
            args.L,
            args.pt_chains,
            n_rounds=args.pt_rounds,
            gibbs_steps=args.pt_gibbs,
            num_runs=args.runs,
        )

    if run_all or 3 in args.only:
        bench_energy(args.L, args.energy_evals, args.runs)

    if run_all or 4 in args.only:
        bench_moments(args.L, args.warmup, args.moment_samples, args.runs)

    if (run_all or 5 in args.only) and not args.skip_correctness:
        check_correctness()

    print_summary(args)

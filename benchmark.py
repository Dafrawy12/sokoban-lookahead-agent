"""
sokoban/benchmark.py

Headless benchmark — runs all 10 levels N times per agent and saves results.

Agents:
  1. astar      — A* solver
  2. pure_llm   — LLM picks every move directly, no heuristic
  3. hybrid     — LLM proposes, heuristic scores, physics verifier filters

Usage:
  python benchmark.py              # 3 runs per level (quick test)
  python benchmark.py --runs 10    # 10 runs per level (full benchmark)
  python benchmark.py --agent astar          # one agent only
  python benchmark.py --levels 1 2 3        # specific levels only
"""

import argparse, json, csv, pathlib, time, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from game   import load_levels, level_to_state
from solver import solve

LEVELS_PATH  = pathlib.Path(__file__).parent / "levels" / "levels.json"
RESULTS_JSON = pathlib.Path(__file__).parent / "results.json"
RESULTS_CSV  = pathlib.Path(__file__).parent / "results.csv"


# ── A* benchmark ─────────────────────────────────────────────────────────────

def run_astar(level) -> dict:
    state  = level_to_state(level)
    result = solve(state)
    return {
        "solved":              result.solved,
        "steps":               len(result.full_path) if result.solved else None,
        "pushes":              result.num_pushes     if result.solved else None,
        "nodes":               result.nodes_expanded,
        "compute_ms":          result.runtime_ms,
        "rejections":          0,
        "llm_proposals_total": None,
        "illegal_rejected":    None,
        "deadlock_rejected":   None,
    }


# ── Pure LLM benchmark ────────────────────────────────────────────────────────

def run_pure_llm(level) -> dict:
    import threading
    from pure_llm_agent import PureLLMAgent

    result   = {}
    done_evt = threading.Event()

    def on_solution(path):
        result["path"] = path
        done_evt.set()
    def on_status(msg):
        pass
    def on_error(msg):
        result["error"] = msg
        done_evt.set()

    agent = PureLLMAgent(on_solution, on_status, on_error)
    agent.start(level_to_state(level))
    done_evt.wait(timeout=300)
    agent.stop()

    solved = "path" in result
    path   = result.get("path", [])

    # Count pushes from path
    pushes = 0
    if solved:
        state = level_to_state(level)
        for direction, _, _ in path:
            ns = state.apply_move(direction)
            if ns and ns.boxes != state.boxes:
                pushes += 1
            if ns:
                state = ns

    return {
        "solved":              solved,
        "steps":               agent.steps_taken if solved else None,
        "pushes":              pushes if solved else None,
        "nodes":               None,
        "compute_ms":          agent.compute_ms,
        "rejections":          agent.rejections,
        "llm_proposals_total": None,
        "illegal_rejected":    None,
        "deadlock_rejected":   None,
    }


# ── Hybrid LLM+ToT benchmark ──────────────────────────────────────────────────

def run_hybrid(level) -> dict:
    import threading
    from llm_tot_agent import LLMToTAgent

    result   = {}
    done_evt = threading.Event()

    def on_solution(path):
        result["path"] = path
        done_evt.set()
    def on_status(msg):
        pass
    def on_error(msg):
        result["error"] = msg
        done_evt.set()

    agent = LLMToTAgent(on_solution, on_status, on_error)
    agent.start(level_to_state(level))
    done_evt.wait(timeout=300)
    agent.stop()

    solved = "path" in result
    path   = result.get("path", [])

    pushes = 0
    if solved:
        state = level_to_state(level)
        for direction, _, _ in path:
            ns = state.apply_move(direction)
            if ns and ns.boxes != state.boxes:
                pushes += 1
            if ns:
                state = ns

    return {
        "solved":              solved,
        "steps":               agent.steps_taken if solved else None,
        "pushes":              pushes if solved else None,
        "nodes":               None,
        "compute_ms":          agent.compute_ms,
        "rejections":          agent.rejections,
        "llm_proposals_total": agent.llm_proposals_total,
        "illegal_rejected":    agent.illegal_rejected,
        "deadlock_rejected":   agent.deadlock_rejected,
    }


# ── Runner ────────────────────────────────────────────────────────────────────

AGENT_FNS = {
    "astar":    run_astar,
    "pure_llm": run_pure_llm,
    "hybrid":   run_hybrid,
}

def run_benchmark(agents, level_ids, runs_per_level):
    levels  = load_levels(LEVELS_PATH)
    levels  = [l for l in levels if l["id"] in level_ids]
    records = []

    total = len(agents) * len(levels) * runs_per_level
    done  = 0

    for agent_name in agents:
        fn = AGENT_FNS[agent_name]
        for level in levels:
            for run in range(1, runs_per_level + 1):
                done += 1
                tag = f"[{done}/{total}] {agent_name} | L{level['id']} {level['name']} | run {run}"
                print(f"\n{tag}")
                sys.stdout.flush()

                t0  = time.perf_counter()
                res = fn(level)
                wall_ms = (time.perf_counter() - t0) * 1000

                record = {
                    "agent":      agent_name,
                    "level_id":   level["id"],
                    "level_name": level["name"],
                    "difficulty": level.get("difficulty", "?"),
                    "run":        run,
                    **res,
                    "wall_ms":    round(wall_ms, 1),
                }
                records.append(record)

                status = "SOLVED" if res["solved"] else "FAILED"
                print(f"  {status}  steps={res['steps']}  pushes={res['pushes']}  "
                      f"compute_ms={res['compute_ms']:.0f}  "
                      f"rejections={res['rejections']}")

    return records


def save_results(records):
    # JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records to {RESULTS_JSON}")

    # CSV
    if records:
        fields = list(records[0].keys())
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)
        print(f"Saved CSV to {RESULTS_CSV}")


def print_summary(records):
    from collections import defaultdict
    print("\n" + "=" * 65)
    print(f"{'BENCHMARK SUMMARY':^65}")
    print("=" * 65)
    print(f"{'Agent':<12} {'Level':<18} {'Runs':<5} {'Solved':<7} "
          f"{'Avg Steps':<11} {'Avg ms':<10} {'Avg Reject'}")
    print("-" * 65)

    # Group by agent + level
    groups = defaultdict(list)
    for r in records:
        groups[(r["agent"], r["level_id"], r["level_name"])].append(r)

    for (agent, lid, lname), runs in sorted(groups.items()):
        solved     = [r for r in runs if r["solved"]]
        n_solved   = len(solved)
        avg_steps  = sum(r["steps"] for r in solved) / n_solved if solved else "-"
        avg_ms     = sum(r["compute_ms"] for r in runs) / len(runs)
        avg_rej    = sum(r["rejections"] for r in runs) / len(runs)
        steps_str  = f"{avg_steps:.1f}" if isinstance(avg_steps, float) else avg_steps
        print(f"{agent:<12} {lname:<18} {len(runs):<5} "
              f"{n_solved}/{len(runs):<5} {steps_str:<11} "
              f"{avg_ms:<10.0f} {avg_rej:.1f}")

    print("=" * 65)

    # ── Topological Fidelity Report (Hybrid only) ─────────────────────────────
    hybrid_runs = [r for r in records if r["agent"] == "hybrid"
                   and r.get("llm_proposals_total") is not None]
    if hybrid_runs:
        total_prop  = sum(r["llm_proposals_total"] for r in hybrid_runs)
        total_ill   = sum(r["illegal_rejected"]    for r in hybrid_runs)
        total_dl    = sum(r["deadlock_rejected"]   for r in hybrid_runs)
        total_valid = total_prop - total_ill - total_dl
        fidelity    = (total_valid / total_prop * 100) if total_prop > 0 else 0.0

        print(f"\n{'TOPOLOGICAL FIDELITY (Hybrid Agent)':^65}")
        print("=" * 65)
        print(f"  Total LLM proposals   : {total_prop}")
        print(f"  Illegal (Phi=0 move)  : {total_ill}  "
              f"({total_ill/total_prop*100:.1f}% of proposals)")
        print(f"  Deadlock (Phi=0 state): {total_dl}  "
              f"({total_dl/total_prop*100:.1f}% of proposals)")
        print(f"  Valid (passed Phi)     : {total_valid}  "
              f"({total_valid/total_prop*100:.1f}% of proposals)")
        print(f"  Fidelity rate         : {fidelity:.1f}%")
        print("=" * 65)

        # Per-level breakdown
        print(f"\n{'Per-Level Topological Fidelity':^65}")
        print("-" * 65)
        print(f"{'Level':<22} {'Proposals':<11} {'Illegal':<10} {'Deadlock':<11} {'Fidelity'}")
        print("-" * 65)
        lv_groups = defaultdict(list)
        for r in hybrid_runs:
            lv_groups[(r["level_id"], r["level_name"])].append(r)
        for (lid, lname), lruns in sorted(lv_groups.items()):
            lp  = sum(r["llm_proposals_total"] for r in lruns)
            li  = sum(r["illegal_rejected"]    for r in lruns)
            ld  = sum(r["deadlock_rejected"]   for r in lruns)
            lv  = lp - li - ld
            lf  = (lv / lp * 100) if lp > 0 else 0.0
            print(f"{lname:<22} {lp:<11} {li:<10} {ld:<11} {lf:.1f}%")
        print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sokoban Benchmark Runner")
    parser.add_argument("--runs",   type=int, default=3,
                        help="Number of runs per level (default 3)")
    parser.add_argument("--agents", nargs="+",
                        choices=["astar","pure_llm","hybrid"],
                        default=["astar","pure_llm","hybrid"],
                        help="Which agents to benchmark")
    parser.add_argument("--levels", nargs="+", type=int,
                        default=list(range(1, 11)),
                        help="Which level IDs to include (default 1-10)")
    args = parser.parse_args()

    print(f"Benchmark: agents={args.agents} levels={args.levels} runs={args.runs}")
    print(f"Total runs: {len(args.agents) * len(args.levels) * args.runs}\n")

    records = run_benchmark(args.agents, args.levels, args.runs)
    save_results(records)
    print_summary(records)

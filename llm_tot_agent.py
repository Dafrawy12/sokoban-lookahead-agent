"""
sokoban/llm_tot_agent.py  —  Hybrid LLM + ToT agent

Simple greedy loop:
  1. LLM sees board → suggests 3 moves
  2. ToT verifies each (apply_move + has_deadlock)
  3. ToT scores survivors (Manhattan, one box focus)
  4. Commit the best scoring move
  5. Repeat from step 1

Backtrack conditions (ONLY two):
  A. Deadlock detected after a move → backtrack before that move
  B. Box-sequence deadlock: placed a box, then got deadlocked later →
     backtrack to before that box was placed, try different path

Loop detection: if same sequence repeated 3x in committed path,
  tell LLM to avoid those moves next ask.

── NEW CONSTRAINT (Phase 2) ────────────────────────────────────────
  Two-Phase Scoring:
  If box_to_target distance has not improved in the last 10 steps,
  switch to Phase 1 mode — ignore box distance, score purely on
  player_to_push_position. This allows the agent to temporarily move
  the box away from the target to maneuver around obstacles.
  Resets back to normal scoring when box distance improves.
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import re, threading, time as _time
from typing import Callable
from game import GameState, TARGET, DIRECTIONS
from solver import has_deadlock, deadlock_type

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen2.5:7b"
MAX_STEPS  = 100   # stop and report unsolved after this many steps


# ── Reflexion memory ──────────────────────────────────────────────────────────

def _add_to_deadlock_memory(memory: list, box_pos, dtype: str | None, push_dir: str | None):
    """
    Append a deadlock event to the reflexion memory list.
    Skips exact duplicates (same box position + same deadlock type).
    Called whenever Phi catches or the agent backtracks from a deadlock.
    """
    if dtype is None:
        dtype = "deadlock"
    key = (box_pos, dtype)
    if not any((m["box_pos"], m["dtype"]) == key for m in memory):
        memory.append({"box_pos": box_pos, "dtype": dtype, "push_dir": push_dir})


# ── Board ─────────────────────────────────────────────────────────────────────

def _board_to_text(state: GameState) -> str:
    col_nums = "   " + " ".join(str(c) for c in range(state.cols))
    rows = [col_nums]
    for r in range(state.rows):
        row = f"{r:2} "
        for c in range(state.cols):
            cell = state.grid[r][c]
            pos  = (r, c)
            if pos == state.player:
                row += "+" if cell == TARGET else "@"
            elif pos in state.boxes:
                row += "*" if cell == TARGET else "$"
            else:
                row += cell
            row += " "
        rows.append(row)
    board_str = "\n".join(rows)
    targets    = [(r,c) for r in range(state.rows) for c in range(state.cols)
                  if state.grid[r][c] == TARGET]
    off_target = [b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET]
    free_tgts  = [t for t in targets if t not in state.boxes]
    return (f"{board_str}\n\n"
            f"Symbols: @ player  $ box  . target  * solved  # wall\n"
            f"Player: row={state.player[0]}, col={state.player[1]}\n"
            f"Boxes not on target: {off_target}\n"
            f"Free targets: {free_tgts}")


def _build_prompt(state, history, excluded=[], focus_box=None, deadlock_memory=None):
    recent = history[-5:] if len(history) > 5 else history
    avoid  = f"\nDo NOT suggest: {', '.join(excluded)} (these are looping)" if excluded else ""

    # Tell LLM exactly which box to focus on and where its target is
    focus_hint = ""
    if focus_box:
        targets   = [(r,c) for r in range(state.rows) for c in range(state.cols)
                     if state.grid[r][c] == TARGET]
        free_tgts = [t for t in targets if t not in state.boxes]
        if free_tgts:
            nearest = min(free_tgts,
                          key=lambda t: abs(focus_box[0]-t[0])+abs(focus_box[1]-t[1]))
            dr = nearest[0]-focus_box[0]
            dc = nearest[1]-focus_box[1]
            sr = (1 if dr>0 else -1) if dr!=0 else 0
            sc = (1 if dc>0 else -1) if dc!=0 else 0
            push_pos = (focus_box[0]-sr, focus_box[1]-sc)
            focus_hint = (f"\nFocus on box at row={focus_box[0]}, col={focus_box[1]}."
                          f"\nIts nearest target is at row={nearest[0]}, col={nearest[1]}."
                          f"\nTo push it there, player must reach row={push_pos[0]}, col={push_pos[1]}.")

    # Reflexion: inject deadlock memory so the LLM avoids previously failed positions
    reflexion_hint = ""
    if deadlock_memory:
        lines = []
        for m in deadlock_memory[-6:]:  # cap at 6 entries to keep prompt short
            if m["box_pos"]:
                r, c = m["box_pos"]
                dir_str = f" via {m['push_dir']}" if m["push_dir"] else ""
                lines.append(
                    f"  - Pushing box to row={r},col={c}{dir_str} "
                    f"caused {m['dtype']} — do NOT repeat this"
                )
            else:
                lines.append(f"  - {m['dtype']} was detected — change your strategy")
        if lines:
            reflexion_hint = (
                "\nDEADLOCK MEMORY — learn from these past failures and avoid repeating them:\n"
                + "\n".join(lines) + "\n"
            )

    return f"""You are helping solve a Sokoban puzzle.

RULES:
- Push boxes ($) onto targets (.)
- Walk player (@) INTO a box to push it one step in that direction
- Cannot push into walls or other boxes

{_board_to_text(state)}
{focus_hint}{reflexion_hint}
Recent moves: {' -> '.join(recent) if recent else 'none'}{avoid}

Suggest exactly 3 DIFFERENT candidate moves to get the focused box onto its target.
All 3 must be different directions.

Reply ONLY in this format:
CANDIDATES: UP, LEFT, DOWN"""


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(prompt):
    import urllib.request, json
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.3, "num_predict": 30}
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()).get("response", "").strip()


def _parse_candidates(text):
    """
    Grab direction words from response. Deduplicates automatically.
    Handles: UP, DOWN, LEFT, RIGHT and variants like PUSHLEFT etc.
    """
    dirs = []
    tokens = re.split(r'[\s,:|.()\-_]+', text.upper())
    for token in tokens:
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            if d in token and d not in dirs:  # not in dirs = no duplicates
                dirs.append(d)
        if len(dirs) == 3:
            break
    return dirs


# ── Scorer: one box at a time ─────────────────────────────────────────────────

def _pick_focus(state) -> tuple | None:
    """Pick the unplaced box closest to any free target."""
    targets    = [(r,c) for r in range(state.rows) for c in range(state.cols)
                  if state.grid[r][c] == TARGET]
    off_target = [b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET]
    free_tgts  = [t for t in targets if t not in state.boxes]
    if not off_target or not free_tgts:
        return None
    return min(off_target,
               key=lambda b: min(abs(b[0]-t[0])+abs(b[1]-t[1]) for t in free_tgts))


def _score(state, focus_box):
    """
    Score based on focused box only, using unique target assignment.
    Primary:  Manhattan(focus_box -> its assigned target).
    Tiebreak: Manhattan(player -> push position for that target).
    Lower = better.
    """
    off_target = [b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET]
    if not off_target: return (0.0, 0.0)

    assignment = _assign_targets(state)
    focus = focus_box if focus_box in off_target else _pick_focus(state)
    if focus is None or focus not in assignment: return (0.0, 0.0)

    nearest = assignment[focus]
    bd = abs(focus[0]-nearest[0]) + abs(focus[1]-nearest[1])
    dr = nearest[0] - focus[0]
    dc = nearest[1] - focus[1]
    sr = (1 if dr > 0 else -1) if dr != 0 else 0
    sc = (1 if dc > 0 else -1) if dc != 0 else 0
    push_pos = (focus[0] - sr, focus[1] - sc)
    pd = abs(state.player[0]-push_pos[0]) + abs(state.player[1]-push_pos[1])
    return (float(bd), float(pd))


# ── NEW CONSTRAINT (Phase 2) ──────────────────────────────────────────────────

def _bfs_distance(state, start, goal):
    """
    BFS distance from start to goal through free cells.
    Treats boxes as obstacles (player can't walk through them).
    Returns Manhattan distance as fallback if goal unreachable.
    """
    if start == goal: return 0
    from collections import deque
    visited = {start}
    queue   = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            npos = (nr, nc)
            if npos == goal: return dist + 1
            if (0 <= nr < state.rows and 0 <= nc < state.cols
                    and npos not in visited
                    and state.grid[nr][nc] != '#'
                    and npos not in state.boxes):
                visited.add(npos)
                queue.append((npos, dist+1))
    # Fallback: Manhattan (goal unreachable — wall or box in way)
    return abs(start[0]-goal[0]) + abs(start[1]-goal[1])


def _score_phase1(state, focus_box):
    """
    Phase 1 scoring — used when no box has been pushed in N steps.
    Primary:  BFS distance from player to push position (accounts for walls/curves).
    Tiebreak: box distance to target.
    Lower = better.
    """
    off_target = [b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET]
    if not off_target: return (0.0, 0.0)

    assignment = _assign_targets(state)
    focus = focus_box if focus_box in off_target else _pick_focus(state)
    if focus is None or focus not in assignment: return (0.0, 0.0)

    nearest  = assignment[focus]
    dr = nearest[0] - focus[0]
    dc = nearest[1] - focus[1]
    sr = (1 if dr > 0 else -1) if dr != 0 else 0
    sc = (1 if dc > 0 else -1) if dc != 0 else 0
    push_pos = (focus[0] - sr, focus[1] - sc)

    # BFS instead of Manhattan — handles walls and curved paths
    pd = _bfs_distance(state, state.player, push_pos)
    bd = abs(focus[0]-nearest[0]) + abs(focus[1]-nearest[1])
    return (float(pd), float(bd))


def _assign_targets(state):
    """
    NEW CONSTRAINT: Greedy unique assignment of boxes to targets.
    Each box is assigned its nearest free target that no closer box has claimed.
    Returns dict: {box_pos: target_pos}
    This ensures each box is scored against its OWN dedicated target,
    preventing two boxes from competing for the same goal.
    """
    targets    = [(r,c) for r in range(state.rows) for c in range(state.cols)
                  if state.grid[r][c] == TARGET]
    off_target = [b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET]
    free_tgts  = [t for t in targets if t not in state.boxes]

    assignment = {}
    remaining  = list(free_tgts)
    # Sort boxes by distance to nearest target — closest box gets first pick
    for box in sorted(off_target,
                      key=lambda b: min((abs(b[0]-t[0])+abs(b[1]-t[1]))
                                        for t in remaining) if remaining else 0):
        if not remaining: break
        best_t = min(remaining, key=lambda t: abs(box[0]-t[0])+abs(box[1]-t[1]))
        assignment[box] = best_t
        remaining.remove(best_t)
    return assignment


def _boxes_remaining(state):
    return len([b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET])


# ── Agent ─────────────────────────────────────────────────────────────────────

class LLMToTAgent:
    def __init__(self, on_solution: Callable, on_status: Callable,
                 on_error: Callable, on_step: Callable = None):
        self.on_solution = on_solution
        self.on_status   = on_status
        self.on_error    = on_error
        self.on_step     = on_step or (lambda d: None)
        self._stop_evt   = threading.Event()
        self._thread     = None
        self.running     = False
        self.steps_taken        = 0
        self.compute_ms         = 0.0
        self.rejections         = 0
        self.backtracks         = 0
        self.nodes              = 0   # LLM calls = nodes explored
        # Topological Fidelity counters
        self.llm_proposals_total = 0  # every direction parsed from LLM
        self.illegal_rejected    = 0  # Phi rejected: physically illegal move
        self.deadlock_rejected   = 0  # Phi rejected: move leads to deadlock
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        except Exception as e:
            raise RuntimeError(f"Ollama not running: {e}")

    def start(self, initial_state: GameState):
        if self.running: return
        self._stop_evt.clear()
        self.running = True; self.steps_taken = 0
        self.backtracks=0; self.compute_ms=0.0; self.rejections=0; self.nodes=0
        self.llm_proposals_total=0; self.illegal_rejected=0; self.deadlock_rejected=0
        self._thread = threading.Thread(
            target=self._solve, args=(initial_state,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set(); self.running = False

    def _ask_llm(self, state, history, excluded=[], last_move=None,
                 focus_box=None, scorer=None, phase1=False, locked_boxes=None,
                 deadlock_memory=None):
        """Ask LLM for 3 candidates, verify+score. Retry up to 5x if < 3."""
        if scorer is None:
            scorer = _score
        if locked_boxes is None:
            locked_boxes = set()
        if deadlock_memory is None:
            deadlock_memory = []
        OPPOSITE = {"UP":"DOWN","DOWN":"UP","LEFT":"RIGHT","RIGHT":"LEFT"}
        came_from = OPPOSITE.get(last_move)  # direction we came FROM (going back = bad)

        for attempt in range(5):
            try:
                prompt = _build_prompt(state, history, excluded, focus_box,
                                       deadlock_memory)
                self.on_status(f"[LLM prompt]\n{'─'*60}\n{prompt}\n{'─'*60}")
                t0  = _time.perf_counter()
                raw = _call_ollama(prompt)
                self.compute_ms += (_time.perf_counter()-t0)*1000
                self.on_status(f"[LLM raw] {raw[:120]}")
                dirs = _parse_candidates(raw)
                self.on_status(f"[LLM parsed] {dirs}")
            except Exception as e:
                self.on_status(f"[LLM error] {e}")
                dirs = []
            if len(dirs) < 3:
                self.on_status(f"[LLM] Only {len(dirs)} dirs — retrying...")
                continue

            # Count every parsed direction as a proposal (before Phi filtering)
            self.llm_proposals_total += len(dirs)

            valid = []
            assignment = _assign_targets(state)
            for d in dirs:
                if d not in DIRECTIONS: continue
                ns = state.apply_move(d)
                if ns is None:
                    self.on_status(f"  Phi({d}): ILLEGAL")
                    self.illegal_rejected += 1
                    self.rejections += 1; continue
                if has_deadlock(ns):
                    self.on_status(f"  Phi({d}): DEADLOCK")
                    self.deadlock_rejected += 1
                    self.rejections += 1
                    # Record this failed position in reflexion memory
                    dtype = deadlock_type(ns)
                    pushed_box = next(iter(ns.boxes - state.boxes), None)
                    _add_to_deadlock_memory(deadlock_memory, pushed_box, dtype, d)
                    continue
                pushed = ns.boxes != state.boxes
                # Determine the correct focus position for scoring
                # If this move pushes the focus box, use its NEW position
                score_focus = focus_box
                if pushed and focus_box:
                    prev_off = set(b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET)
                    cur_off  = set(b for b in ns.boxes   if ns.grid[b[0]][b[1]] != TARGET)
                    moved    = cur_off - prev_off
                    if moved and focus_box not in cur_off:
                        score_focus = next(iter(moved))  # box moved here
                # Phase 1 scorer only for repositioning, normal scorer for pushes
                effective_scorer = _score if pushed else scorer
                score = effective_scorer(ns, score_focus)
                if pushed and focus_box and focus_box in assignment:
                    if not (pushed and focus_box):  # already computed above
                        prev_off = set(b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET)
                        cur_off  = set(b for b in ns.boxes   if ns.grid[b[0]][b[1]] != TARGET)
                        moved    = cur_off - prev_off

                    # Hard reject: never push a solved box off its target
                    if locked_boxes and moved:
                        solved_before = set(b for b in state.boxes if state.grid[b[0]][b[1]] == TARGET)
                        solved_after  = set(b for b in ns.boxes   if ns.grid[b[0]][b[1]] == TARGET)
                        if len(solved_after) < len(solved_before):
                            self.on_status(f"  Phi({d}): LOCKED BOX — REJECTED")
                            self.rejections += 1; continue

                    # Score the push direction
                    reason = ""
                    if moved:
                        new_pos  = next(iter(moved))
                        tgt      = assignment[focus_box]
                        old_dist = abs(focus_box[0]-tgt[0]) + abs(focus_box[1]-tgt[1])
                        new_dist = abs(new_pos[0]-tgt[0])   + abs(new_pos[1]-tgt[1])
                        if len(cur_off) < len(prev_off):
                            score = (-1.0, -1.0);  reason = "GOAL PUSH"
                        elif new_dist > old_dist:
                            if phase1:
                                score = (score[0] - 0.5, score[1] - 0.5); reason = "push-ph1"
                            else:
                                score = (999.0, 999.0);  reason = "WASTED PUSH"
                        else:
                            score = (score[0] - 0.5, score[1] - 0.5)
                    else:
                        score = (-1.0, -1.0);  reason = "GOAL PUSH"
                elif pushed:
                    score = (score[0] - 0.5, score[1] - 0.5)
                    reason = "push"
                else:
                    # Walk step — penalise if it moves the player AWAY from the push position
                    reason = "walk"
                    if focus_box and focus_box in assignment:
                        tgt_w      = assignment[focus_box]
                        dr_w       = tgt_w[0] - focus_box[0]
                        dc_w       = tgt_w[1] - focus_box[1]
                        sr_w       = (1 if dr_w > 0 else -1) if dr_w != 0 else 0
                        sc_w       = (1 if dc_w > 0 else -1) if dc_w != 0 else 0
                        push_pos_w = (focus_box[0] - sr_w, focus_box[1] - sc_w)
                        old_pd     = abs(state.player[0] - push_pos_w[0]) + abs(state.player[1] - push_pos_w[1])
                        new_pd     = abs(ns.player[0]    - push_pos_w[0]) + abs(ns.player[1]    - push_pos_w[1])
                        if new_pd > old_pd:
                            score = (score[0] + 10.0, score[1] + 10.0)
                            reason = "walk↑AWAY"
                is_reverse = (d == came_from)
                if is_reverse: reason += "(rev)"
                self.on_status(f"  Phi({d}): VALID [{reason}] box={score[0]:.1f} pl={score[1]:.1f}")
                valid.append({"move": d, "state": ns, "score": score,
                              "pushed": pushed, "reverse": is_reverse})
            if valid:
                # Sort by score first, then penalise going back where we came from
                valid.sort(key=lambda v: (v["score"], 1 if v["reverse"] else 0))
                return valid
            self.on_status("[ToT] All rejected — retrying LLM...")
        return []

    def _detect_loop(self, path):
        """If last moves form a repeating pattern 3x, return moves to exclude."""
        if len(path) < 6: return []
        tail = path[-12:]
        for plen in (2, 3, 4):
            if len(tail) >= plen * 3:
                p = tail[-plen:]
                if tail[-plen*2:-plen] == p and tail[-plen*3:-plen*2] == p:
                    return list(set(p))
        return []

    def _solve(self, initial_state: GameState):
        def log(msg): self.on_status(msg)
        try:
            state   = initial_state
            history = []
            states  = [initial_state]
            path    = []
            n_total = _boxes_remaining(initial_state)

            # Lock onto one box at a time — switch only when it's placed
            focus_box    = _pick_focus(initial_state)
            locked_boxes = set()

            # Scale stagnation limit by map size — larger maps need more steps to navigate
            STAGNATION_LIMIT = max(6, (initial_state.rows + initial_state.cols) // 2)
            log(f"Hybrid LLM+ToT starting... {n_total} box(es) | focus: {focus_box} | stagnation limit: {STAGNATION_LIMIT}")

            # ── Two-phase scoring state ───────────────────────────────────────
            steps_since_push = 0
            phase1_mode      = False

            # ── Reflexion memory — persists across the whole level run ────────
            deadlock_memory: list = []

            while not state.is_solved() and not self._stop_evt.is_set():
                if self.steps_taken >= MAX_STEPS:
                    self.on_error(f"Reached {MAX_STEPS} step limit"); return

                log(f"--- Step {self.steps_taken+1} | focus={focus_box} "
                    f"{'[PHASE 1 - reposition]' if phase1_mode else '[PHASE 2 - push]'} ---")

                # ── NEW: stagnation check — track steps WITHOUT a push ────────
                steps_since_push += 1
                if steps_since_push >= STAGNATION_LIMIT and not phase1_mode:
                    phase1_mode      = True
                    steps_since_push = 0
                    log(f"[Phase] {STAGNATION_LIMIT} steps without a push "
                        f"— switching to Phase 1 (reposition mode)")
                # ─────────────────────────────────────────────────────────────

                # ── Loop detection ────────────────────────────────────────────
                excluded = self._detect_loop(history)
                if excluded:
                    log(f"[Loop] Avoiding {excluded}")

                # Ask LLM + verify + score
                scorer = _score_phase1 if phase1_mode else _score
                candidates = self._ask_llm(state, history, excluded,
                                           last_move=history[-1] if history else None,
                                           focus_box=focus_box,
                                           scorer=scorer,
                                           phase1=phase1_mode,
                                           locked_boxes=locked_boxes,
                                           deadlock_memory=deadlock_memory)
                self.nodes += 1

                if not candidates:
                    log("[Stuck] No valid moves — backtracking...")
                    # Reflexion: record deadlock type of the current dead-end state
                    dtype = deadlock_type(state)
                    if dtype:
                        last_pushed = (next(iter(states[-1].boxes - states[-2].boxes), None)
                                       if len(states) >= 2 else None)
                        _add_to_deadlock_memory(deadlock_memory, last_pushed, dtype,
                                                history[-1] if history else None)
                        log(f"[Reflexion] Recorded: {dtype} at {last_pushed}")
                    state, history, states, path = \
                        self._backtrack(state, history, states, path, n_total, log)
                    if state is None:
                        self.on_error("Completely stuck — no solution found"); return
                    focus_box = _pick_focus(state)
                    locked_boxes = set(b for b in state.boxes
                                       if state.grid[b[0]][b[1]] == TARGET)
                    steps_since_push = 0; phase1_mode = False
                    continue

                # Take the best scoring move
                best = candidates[0]
                ns   = best["state"]

                if has_deadlock(ns):
                    log(f"[Deadlock] {best['move']} — backtracking...")
                    # Reflexion: record why this candidate state is a deadlock
                    dtype = deadlock_type(ns)
                    pushed_box = next(iter(ns.boxes - state.boxes), None)
                    _add_to_deadlock_memory(deadlock_memory, pushed_box, dtype, best["move"])
                    log(f"[Reflexion] Recorded: {dtype} at {pushed_box}")
                    state, history, states, path = \
                        self._backtrack(state, history, states, path, n_total, log)
                    if state is None:
                        self.on_error("Completely stuck — no solution found"); return
                    focus_box = _pick_focus(state)
                    locked_boxes = set(b for b in state.boxes if state.grid[b[0]][b[1]] == TARGET)
                    steps_since_push = 0; phase1_mode = False
                    continue

                # Commit
                prev_boxes_set = set(b for b in state.boxes
                                     if state.grid[b[0]][b[1]] != TARGET)
                history.append(best["move"])
                states.append(ns)
                path.append((best["move"], [], "hybrid"))
                state = ns
                self.steps_taken += 1

                cur_boxes_set = set(b for b in ns.boxes
                                    if ns.grid[b[0]][b[1]] != TARGET)

                # Reset push counter and phase on ANY push
                if best["pushed"]:
                    steps_since_push = 0
                    if phase1_mode:
                        phase1_mode = False
                        log(f"  [Phase] Box pushed — back to Phase 2 (push mode)")

                # Track focus_box as it moves (follow the push)
                # but ONLY switch focus when it actually lands on a target
                if focus_box and focus_box not in cur_boxes_set:
                    moved = cur_boxes_set - prev_boxes_set
                    if moved:
                        focus_box = next(iter(moved))
                        log(f"  [Focus follows push] -> {focus_box}")
                    else:
                        # Box landed on target — lock it, pick nearest to player next
                        old_focus = focus_box
                        locked_boxes.add(old_focus)
                        # NEW: pick nearest unplaced box to player (not just nearest to target)
                        targets    = [(r,c) for r in range(state.rows)
                                      for c in range(state.cols) if state.grid[r][c] == TARGET]
                        off_target = [b for b in state.boxes
                                      if state.grid[b[0]][b[1]] != TARGET
                                      and b not in locked_boxes]
                        if off_target:
                            focus_box = min(off_target,
                                           key=lambda b: abs(b[0]-state.player[0])
                                                        +abs(b[1]-state.player[1]))
                        else:
                            focus_box = _pick_focus(state)
                        locked_boxes = set(b for b in state.boxes if state.grid[b[0]][b[1]] == TARGET)
                        steps_since_push = 0; phase1_mode = False
                        log(f"  ** BOX ON TARGET ({old_focus}) → new focus (nearest to player): {focus_box} **")

                placed = " ** BOX ON TARGET **" if len(cur_boxes_set) < len(prev_boxes_set) else ""
                log(f"  -> {best['move']} box={best['score'][0]:.1f} pl={best['score'][1]:.1f}{placed}")

                self.on_step(("PATH", [p[0] for p in path]))

                # Post-commit deadlock check — catches configuration deadlocks
                # that only become visible after the move is committed
                if has_deadlock(state):
                    log(f"[Post-commit Deadlock] State is deadlocked — backtracking...")
                    # Reflexion: record the committed move that created this deadlock
                    dtype = deadlock_type(state)
                    last_pushed = (next(iter(states[-1].boxes - states[-2].boxes), None)
                                   if len(states) >= 2 else None)
                    _add_to_deadlock_memory(deadlock_memory, last_pushed, dtype,
                                            history[-1] if history else None)
                    log(f"[Reflexion] Recorded post-commit: {dtype} at {last_pushed}")
                    state, history, states, path = \
                        self._backtrack(state, history, states, path, n_total, log)
                    if state is None:
                        self.on_error("Completely stuck — no solution found"); return
                    focus_box = _pick_focus(state)
                    locked_boxes = set(b for b in state.boxes if state.grid[b[0]][b[1]] == TARGET)
                    steps_since_push = 0; phase1_mode = False

            if state.is_solved():
                log(f"SOLVED in {self.steps_taken} steps ({self.compute_ms:.0f}ms)")
                self.on_solution(path)
            else:
                self.on_error("Stopped")

        except Exception as e:
            self.on_error(f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.running = False

    def _backtrack(self, state, history, states, path, n_total, log):
        """
        Backtrack one step. If a box was placed and we're stuck,
        unwind to before that box was placed.
        Returns (new_state, history, states, path) or (None,...) if can't.
        """
        if len(history) < 1:
            return None, history, states, path

        cur_boxes = _boxes_remaining(state)

        if cur_boxes < n_total:
            # A box is placed and we're stuck — unwind to before that placement
            target = len(history) - 1
            while target > 0:
                prev_boxes = _boxes_remaining(states[target])
                if prev_boxes > cur_boxes:
                    break
                target -= 1
            log(f"[Box-order backtrack] Unwinding from {len(history)} to {target}")
            history = history[:target]
            states  = states[:target+1]
            path    = path[:target]
        else:
            history.pop()
            states.pop()
            path.pop()
            log(f"[Backtrack] Now at depth {len(history)}")

        if not states:
            return None, history, states, path

        self.on_step(("BACKTRACK", [p[0] for p in path]))
        self.backtracks += 1
        return states[-1], history, states, path

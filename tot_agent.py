"""
sokoban/tot_agent.py
Heuristic Tree of Thoughts agent.

Solves the level completely first (pure computation, no animation waits),
then posts the full solution path to the main thread via on_solution().
The main thread animates it exactly like the LLM agent.

Heuristic:
  1. Greedy unique box-target assignment (no two boxes chase same target)
  2. Player-to-nearest-box distance      (keeps player near work)
  3. Push-readiness bonus                (player behind box facing target)
  4. Push-move bonus                     (candidate that pushes scores extra)
  5. 2-step lookahead                    (score best reachable child)
  6. Backtrack memory                    (penalise known dead-end configs)
"""

from __future__ import annotations
import threading
from typing import Optional, Callable
from game import GameState, DIRECTIONS, TARGET
from solver import has_deadlock


# ── Greedy box-target assignment ─────────────────────────────────────────────

def _assign_boxes_to_targets(boxes: list, targets: list) -> dict:
    remaining = list(targets)
    assignment = {}
    sorted_boxes = sorted(
        boxes,
        key=lambda b: min(abs(b[0]-t[0])+abs(b[1]-t[1]) for t in remaining)
        if remaining else 0
    )
    for b in sorted_boxes:
        if not remaining:
            break
        best = min(remaining, key=lambda t: abs(b[0]-t[0])+abs(b[1]-t[1]))
        assignment[b] = best
        remaining.remove(best)
    return assignment


# ── Heuristic ─────────────────────────────────────────────────────────────────

def _score_state(state: GameState,
                 bad_box_positions: set | None = None) -> float:
    targets = [(r,c) for r in range(state.rows)
                     for c in range(state.cols)
                     if state.grid[r][c] == TARGET]
    boxes_remaining = [b for b in state.boxes
                       if state.grid[b[0]][b[1]] != TARGET]
    boxes_solved    = len(state.boxes) - len(boxes_remaining)
    free_targets    = [t for t in targets if t not in state.boxes]

    if not boxes_remaining:
        return float('inf')

    assignment     = _assign_boxes_to_targets(boxes_remaining, free_targets)
    total_box_dist = sum(abs(b[0]-t[0])+abs(b[1]-t[1])
                         for b,t in assignment.items())

    pr, pc = state.player
    player_to_box = min(abs(pr-b[0])+abs(pc-b[1]) for b in boxes_remaining)

    push_bonus = 0
    for b, t in assignment.items():
        dr = t[0]-b[0]; dc = t[1]-b[1]
        sr = (1 if dr>0 else -1) if dr!=0 else 0
        sc = (1 if dc>0 else -1) if dc!=0 else 0
        if state.player == (b[0]-sr, b[1]-sc):
            push_bonus += 4

    bad_penalty = 0
    if bad_box_positions:
        if frozenset(boxes_remaining) in bad_box_positions:
            bad_penalty = 20

    return (
        -total_box_dist * 2
        - 0.5 * player_to_box
        + boxes_solved * 100
        + push_bonus
        - bad_penalty
    )


def _lookahead_score(state: GameState,
                     bad_box_positions: set | None = None) -> float:
    best = _score_state(state, bad_box_positions)
    for d in ('UP','DOWN','LEFT','RIGHT'):
        ns = state.apply_move(d)
        if ns is None or has_deadlock(ns):
            continue
        s = _score_state(ns, bad_box_positions)
        if s > best:
            best = s
    return best


def _evaluate_candidates(state: GameState,
                         bad_box_positions: set | None = None) -> list[dict]:
    candidates = []
    for direction in ('UP','DOWN','LEFT','RIGHT'):
        ns = state.apply_move(direction)
        if ns is None:
            candidates.append({'move':direction,'status':'illegal',
                                'score':None,'detail':'Blocked'})
            continue
        if has_deadlock(ns):
            candidates.append({'move':direction,'status':'deadlock',
                                'score':None,'detail':'Creates deadlock'})
            continue

        pushed          = ns.boxes != state.boxes
        base            = _lookahead_score(ns, bad_box_positions)
        push_move_bonus = 5 if pushed else 0
        score           = base + push_move_bonus

        candidates.append({
            'move': direction, 'status': 'valid', 'score': score,
            'next_state': ns,
            'detail': f'{"PUSH " if pushed else ""}score:{score:.1f}',
        })

    candidates.sort(key=lambda c: (0 if c['status']=='valid' else 1,
                                    -(c['score'] or -9999)))
    return candidates


# ── Tree node ─────────────────────────────────────────────────────────────────

class TreeNode:
    __slots__ = ('state','move','parent','children_tried','candidates','depth')
    def __init__(self, state, move, parent, depth):
        self.state          = state
        self.move           = move
        self.parent         = parent
        self.children_tried = 0
        self.candidates     = []
        self.depth          = depth


# ── Agent ─────────────────────────────────────────────────────────────────────

class ToTAgent:
    """
    Heuristic Tree of Thoughts agent.

    Solves the level entirely in a background thread (no animation waits),
    then delivers the complete path via on_solution().

    Callbacks:
      on_solution(path)  — list of (direction, candidates) once solved
      on_status(msg)     — progress updates while solving
      on_error(msg)      — called if no solution found or limit reached
    """

    MAX_NODES = 50000

    def __init__(self,
                 on_solution: Callable[[list], None],
                 on_status:   Callable[[str], None],
                 on_error:    Callable[[str], None]):
        self.on_solution = on_solution
        self.on_status   = on_status
        self.on_error    = on_error

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.running         = False
        self.nodes_explored  = 0
        self.backtracks      = 0
        self.current_depth   = 0
        self.compute_ms      = 0.0

    def start(self, initial_state: GameState):
        if self.running:
            return
        self._stop_evt.clear()
        self.running        = True
        self.nodes_explored = 0
        self.backtracks     = 0
        self.current_depth  = 0
        self.compute_ms     = 0.0
        self._thread = threading.Thread(
            target=self._run, args=(initial_state,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self.running = False

    def _run(self, initial: GameState):
        import time as _t
        try:
            self.compute_ms     = 0.0
            bad_box_positions: set = set()

            t0 = _t.perf_counter()
            root = TreeNode(initial, None, None, 0)
            root.candidates = _evaluate_candidates(initial, bad_box_positions)
            self.compute_ms += (_t.perf_counter()-t0)*1000

            stack:   list = [root]
            visited: set  = {(initial.player, initial.boxes)}

            # solution_stack tracks the forward path (no backtracks)
            # We reconstruct it from the DFS stack when solved
            while stack and not self._stop_evt.is_set():
                if self.nodes_explored >= self.MAX_NODES:
                    self.on_error(f'Limit reached ({self.MAX_NODES} nodes)')
                    return

                node  = stack[-1]
                state = node.state

                self.current_depth = node.depth
                self.on_status(
                    f'Solving... depth:{node.depth} '
                    f'nodes:{self.nodes_explored} '
                    f'backs:{self.backtracks}'
                )

                if state.is_solved():
                    # Reconstruct the solution path from the DFS stack
                    # (stack contains root → ... → solved node)
                    path = []
                    for n in stack[1:]:   # skip root (no move)
                        path.append((n.move, n.candidates))

                    self.on_status(
                        f'Solved!  Steps:{len(path)}  '
                        f'Nodes:{self.nodes_explored}  '
                        f'Backtracks:{self.backtracks}  '
                        f'Compute:{self.compute_ms:.0f}ms'
                    )
                    self.on_solution(path)
                    return

                valid = [c for c in node.candidates if c['status']=='valid']

                if node.children_tried >= len(valid):
                    # Dead end — record and backtrack
                    boxes_rem = frozenset(b for b in state.boxes
                                          if state.grid[b[0]][b[1]] != TARGET)
                    if boxes_rem:
                        bad_box_positions.add(boxes_rem)
                    stack.pop()
                    if not stack:
                        self.on_error('No solution found')
                        return
                    self.backtracks += 1
                    continue

                # Commit to next best candidate
                candidate = valid[node.children_tried]
                node.children_tried += 1
                self.nodes_explored += 1

                ns  = candidate['next_state']
                key = (ns.player, ns.boxes)
                if key in visited:
                    continue
                visited.add(key)

                t0    = _t.perf_counter()
                child = TreeNode(ns, candidate['move'], node, node.depth+1)
                child.candidates = _evaluate_candidates(ns, bad_box_positions)
                self.compute_ms += (_t.perf_counter()-t0)*1000

                stack.append(child)

        except Exception as e:
            self.on_error(f'Error: {e}')
            import traceback; traceback.print_exc()
        finally:
            self.running = False

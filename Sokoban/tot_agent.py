"""
sokoban/tot_agent.py
Tree of Thoughts agent for Sokoban.

Algorithm: Depth-first search with beam branching.

At each node in the tree:
  1. GENERATE — produce all valid candidate moves (up to 4)
  2. EVALUATE — score each with a heuristic (box-to-target distance + penalties)
  3. SELECT   — commit to the highest-scoring candidate (greedy)
  4. BACKTRACK — if we hit a deadlock or revisited state, undo and try the next branch

This is structurally identical to what an LLM-based ToT would do:
  - Generate = LLM proposes candidates
  - Evaluate = LLM scores each with reasoning
  - Select   = LLM picks best
  - Backtrack = LLM says "this path is wrong, let me reconsider"

The agent runs in a background thread and posts moves to the main thread
one at a time so they can be animated.
"""

from __future__ import annotations
import threading
import time
from typing import Optional, Callable
from game import GameState, DIRECTIONS, TARGET
from solver import has_deadlock


# ── Heuristic ─────────────────────────────────────────────────────────────────

def _score_state(state: GameState) -> float:
    """
    Score a state. Higher = better.
    Uses negative Manhattan distance so closer = higher score.
    Adds a bonus for each box already on a target.
    """
    targets = [(r, c) for r in range(state.rows)
                       for c in range(state.cols)
                       if state.grid[r][c] == TARGET]

    boxes_remaining = [b for b in state.boxes
                       if state.grid[b[0]][b[1]] != TARGET]
    boxes_solved    = len(state.boxes) - len(boxes_remaining)
    free_targets    = [t for t in targets
                       if t not in state.boxes]

    if not boxes_remaining:
        return float('inf')   # solved

    # Sum of each box's distance to its nearest free target
    total_dist = 0
    for b in boxes_remaining:
        if free_targets:
            d = min(abs(b[0]-t[0]) + abs(b[1]-t[1]) for t in free_targets)
            total_dist += d

    return -total_dist + boxes_solved * 100


def _evaluate_candidates(state: GameState) -> list[dict]:
    """
    Generate and score all valid moves from this state.
    Returns list of candidate dicts sorted best-first.
    """
    candidates = []
    for direction in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        next_state = state.apply_move(direction)
        if next_state is None:
            candidates.append({
                'move': direction,
                'status': 'illegal',
                'score': None,
                'detail': 'Blocked',
            })
            continue

        if has_deadlock(next_state):
            candidates.append({
                'move': direction,
                'status': 'deadlock',
                'score': None,
                'detail': 'Creates deadlock',
            })
            continue

        score = _score_state(next_state)
        candidates.append({
            'move': direction,
            'status': 'valid',
            'score': score,
            'next_state': next_state,
            'detail': f'Score: {score:.0f}',
        })

    # Sort: valid moves first, then by score descending
    candidates.sort(key=lambda c: (
        0 if c['status'] == 'valid' else 1,
        -(c['score'] or -9999)
    ))

    return candidates


# ── Tree node ─────────────────────────────────────────────────────────────────

class TreeNode:
    """One node in the ToT search tree."""
    __slots__ = ('state', 'move', 'parent', 'children_tried', 'candidates', 'depth')

    def __init__(self, state: GameState, move: Optional[str],
                 parent: Optional['TreeNode'], depth: int):
        self.state          = state
        self.move           = move          # move that led here
        self.parent         = parent
        self.children_tried = 0            # how many candidates have been tried
        self.candidates: list[dict] = []   # evaluated candidates, best-first
        self.depth          = depth


# ── Agent ─────────────────────────────────────────────────────────────────────

class ToTAgent:
    """
    Tree of Thoughts agent running in a background thread.

    Communication with main thread:
      on_step(move, candidates, backtracking) — main thread applies move + animates
      on_status(msg)                          — UI status line update
      signal_move_done()                      — main thread calls after animation
    """

    MAX_DEPTH   = 500    # max tree depth before giving up
    MAX_NODES   = 50000  # safety limit on total nodes explored
    STEP_DELAY  = 0.0    # extra delay between moves (0 = as fast as animation allows)

    def __init__(self,
                 on_step:   Callable[[str, list[dict], bool], None],
                 on_status: Callable[[str], None]):
        self.on_step   = on_step
        self.on_status = on_status

        self._stop_evt  = threading.Event()
        self._move_done = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.running        = False
        self.nodes_explored = 0
        self.backtracks     = 0
        self.current_depth  = 0
        self.compute_ms     = 0.0
        self.current_candidates: list[dict] = []
        self.is_backtracking = False

    def start(self, get_state: Callable[[], GameState]):
        if self.running:
            return
        self._stop_evt.clear()
        self._move_done.clear()
        self.running         = True
        self.nodes_explored  = 0
        self.backtracks      = 0
        self.current_depth   = 0
        self.compute_ms      = 0.0
        self.is_backtracking = False
        self._thread = threading.Thread(
            target=self._run, args=(get_state,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._move_done.set()
        self.running = False

    def signal_move_done(self):
        """Call from main thread once animation finishes."""
        self._move_done.set()

    # ── main search ───────────────────────────────────────────────────────

    def _run(self, get_state: Callable[[], GameState]):
        import time as _time
        try:
            self.compute_ms = 0.0   # pure algorithm time, no animation waits

            initial = get_state()
            root = TreeNode(initial, None, None, 0)

            t0 = _time.perf_counter()
            root.candidates = _evaluate_candidates(initial)
            self.compute_ms += (_time.perf_counter() - t0) * 1000

            stack:   list[TreeNode]    = [root]
            visited: set[tuple]        = {(initial.player, initial.boxes)}
            path:    list[TreeNode]    = [root]

            while stack and not self._stop_evt.is_set():

                if self.nodes_explored >= self.MAX_NODES:
                    self.on_status(f'Limit reached ({self.MAX_NODES} nodes)')
                    break

                node  = stack[-1]
                state = node.state

                if state.is_solved():
                    self.on_status(f'Solved!  Depth:{node.depth}  Backtracks:{self.backtracks}')
                    break

                valid = [c for c in node.candidates if c['status'] == 'valid']

                if node.children_tried >= len(valid):
                    # All branches exhausted — backtrack
                    stack.pop()
                    if not stack:
                        self.on_status('No solution found')
                        break
                    if node.move:
                        self.backtracks      += 1
                        self.is_backtracking  = True
                        self.on_status(
                            f'Backtracking... depth {node.depth} exhausted, '
                            f'{self.backtracks} backtracks'
                        )
                        self._emit_move(node.move, [], backtracking=True)
                    continue

                # Commit to next best candidate
                candidate = valid[node.children_tried]
                node.children_tried += 1
                self.nodes_explored += 1

                next_state = candidate['next_state']
                key = (next_state.player, next_state.boxes)
                if key in visited:
                    continue
                visited.add(key)

                # Build child and evaluate its candidates (this is the compute work)
                t0 = _time.perf_counter()
                child = TreeNode(next_state, candidate['move'], node, node.depth + 1)
                child.candidates = _evaluate_candidates(next_state)
                self.compute_ms += (_time.perf_counter() - t0) * 1000

                stack.append(child)
                path.append(child)

                self.current_depth      = child.depth
                self.is_backtracking    = False
                self.current_candidates = child.candidates

                self.on_status(
                    f'Depth:{child.depth}  '
                    f'Nodes:{self.nodes_explored}  '
                    f'Backs:{self.backtracks}'
                )
                # Emit move to main thread and WAIT for animation — but don't count the wait
                self._emit_move(candidate['move'], child.candidates, backtracking=False)

        except Exception as e:
            self.on_status(f'Error: {e}')
            import traceback; traceback.print_exc()
        finally:
            self.running = False

    def _emit_move(self, direction: str, candidates: list[dict], backtracking: bool):
        """Send move to main thread and wait for animation."""
        self._move_done.clear()
        self.on_step(direction, candidates, backtracking)
        self._move_done.wait(timeout=10.0)

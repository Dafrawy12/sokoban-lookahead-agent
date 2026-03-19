"""
sokoban/solver.py
Standard step-based A* solver for Sokoban.

State    : (player_pos, boxes) — exact player cell tracked.
Action   : one step in any of 4 directions (walk or push).
Cost     : 1 per step.
Heuristic: sum of each box's Manhattan distance to its nearest target.
Deadlock : multi-type detection — corner, freeze, line.

SolverResult.full_path  — every player step (walk + push), used for playback.
SolverResult.push_sequence — push-only events, used for the overlay.
"""

from __future__ import annotations
import heapq, time
from dataclasses import dataclass
from typing import Optional
from game import GameState, DIRECTIONS, TARGET


@dataclass
class MoveStep:
    """One individual player step — either a walk or a push."""
    direction:   str
    is_push:     bool
    box_from:    Optional[tuple[int,int]]   # None if walk
    box_to:      Optional[tuple[int,int]]
    state_after: GameState


@dataclass
class PushStep:
    """Push-only event — used by the path overlay."""
    direction:     str
    box_from:      tuple[int,int]
    box_to:        tuple[int,int]
    player_before: tuple[int,int]
    player_after:  tuple[int,int]
    state_after:   GameState


@dataclass
class SolverResult:
    solved:          bool
    full_path:       list[MoveStep]   # every step (walk + push) for playback
    push_sequence:   list[PushStep]   # push-only for overlay
    nodes_expanded:  int
    nodes_generated: int
    runtime_ms:      float
    num_pushes:      int

    def summary(self) -> str:
        if self.solved:
            return (f"Solved in {self.num_pushes} pushes | "
                    f"{len(self.full_path)} total steps | "
                    f"{self.nodes_expanded} nodes expanded | "
                    f"{self.runtime_ms:.1f} ms")
        return (f"No solution found | "
                f"{self.nodes_expanded} nodes expanded | "
                f"{self.runtime_ms:.1f} ms")


# ── Deadlock detection ────────────────────────────────────────────────────────
#
# Four independent checks. Any one returning True → state is a deadlock.
#
# 0. DEAD ZONE  — pre-computed per level. Every cell a box can never leave
#                 because the player cannot get behind it to push it out.
#                 Computed once via reverse push flood-fill from all targets.
# 1. CORNER     — box at an L-corner with no targets along either open wall.
# 2. FREEZE     — box (or cluster) frozen on both axes by walls + frozen neighbours.
# 3. LINE       — wall-trapped row/column has more boxes than targets.


# ── Dead-zone table ───────────────────────────────────────────────────────────

def _compute_dead_zones(state: GameState) -> frozenset[tuple[int, int]]:
    """
    Reverse push flood-fill from every target cell.

    Forward push: player at (br-dr, bc-dc) pushes box at (br, bc)
                  → box lands at (br+dr, bc+dc).

    Reverse: given box at destination (br, bc) pushed in direction (dr, dc),
             the box came FROM (br-dr, bc-dc)
             and the player was at  (br-2*dr, bc-2*dc).

    We flood-fill backwards from every target, collecting all cells a box
    could have been pushed FROM to eventually reach a target.
    Any floor cell not collected is a dead zone — no push sequence exists
    that moves a box from there to any target.
    """
    rows, cols = state.rows, state.cols
    grid = state.grid

    reachable: set[tuple[int, int]] = set()
    queue: list[tuple[int, int]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == TARGET:
                reachable.add((r, c))
                queue.append((r, c))

    while queue:
        br, bc = queue.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            prev_br, prev_bc   = br - dr,     bc - dc      # box came from here
            player_r, player_c = br - 2 * dr, bc - 2 * dc  # player was here

            if not (0 <= prev_br < rows and 0 <= prev_bc < cols):
                continue
            if not (0 <= player_r < rows and 0 <= player_c < cols):
                continue
            if grid[prev_br][prev_bc] == '#':
                continue
            if grid[player_r][player_c] == '#':
                continue

            if (prev_br, prev_bc) not in reachable:
                reachable.add((prev_br, prev_bc))
                queue.append((prev_br, prev_bc))

    dead: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != '#' and (r, c) not in reachable:
                dead.add((r, c))

    return frozenset(dead)


_dead_zone_cache: dict[int, frozenset] = {}

def _get_dead_zones(state: GameState) -> frozenset[tuple[int, int]]:
    key = id(state.grid)
    if key not in _dead_zone_cache:
        _dead_zone_cache[key] = _compute_dead_zones(state)
    return _dead_zone_cache[key]


def _dead_zone_deadlock(state: GameState) -> bool:
    """True if any unsolved box sits on a dead-zone cell."""
    dead = _get_dead_zones(state)
    if not dead:
        return False
    for br, bc in state.boxes:
        if state.grid[br][bc] == TARGET:
            continue
        if (br, bc) in dead:
            return True
    return False


def _corner_deadlock(state: GameState, box: tuple[int, int]) -> bool:
    """
    Corner deadlock: a box sits at an L-corner formed by two perpendicular
    walls AND there are no targets accessible by sliding along either wall.

    A box in an L-corner (e.g. wall above + wall left) can only be pushed
    RIGHT or DOWN. It will hug either the top wall (moving right in row r)
    or the left wall (moving down in col c). If neither of those
    wall-hugging paths contains a target, the box can never be solved.

    This avoids the false positive of a 3-sided corridor with a target
    at the open end.
    """
    r, c = box
    if state.grid[r][c] == TARGET:
        return False

    wu = state.is_wall(r - 1, c)
    wd = state.is_wall(r + 1, c)
    wl = state.is_wall(r, c - 1)
    wr = state.is_wall(r, c + 1)

    def targets_in_row(row, col_start, col_step):
        """Any target in the row walking from col_start in col_step direction?"""
        col = col_start
        while 0 <= col < state.cols and not state.is_wall(row, col):
            if state.grid[row][col] == TARGET:
                return True
            col += col_step
        return False

    def targets_in_col(col, row_start, row_step):
        """Any target in the column walking from row_start in row_step direction?"""
        row = row_start
        while 0 <= row < state.rows and not state.is_wall(row, col):
            if state.grid[row][col] == TARGET:
                return True
            row += row_step
        return False

    # Check each L-corner configuration
    corners = []
    if wu and wl:   # top-left corner → can push right (along top wall) or down (along left wall)
        corners.append((targets_in_row(r, c, +1), targets_in_col(c, r, +1)))
    if wu and wr:   # top-right corner → can push left or down
        corners.append((targets_in_row(r, c, -1), targets_in_col(c, r, +1)))
    if wd and wl:   # bottom-left corner → can push right or up
        corners.append((targets_in_row(r, c, +1), targets_in_col(c, r, -1)))
    if wd and wr:   # bottom-right corner → can push left or up
        corners.append((targets_in_row(r, c, -1), targets_in_col(c, r, -1)))

    # Deadlock if in at least one corner AND neither open axis has a target
    return any(not row_ok and not col_ok for row_ok, col_ok in corners)


def _is_frozen(box: tuple[int, int],
               state: GameState,
               memo: dict,
               in_progress: set) -> bool:
    """
    Freeze deadlock — recursive.
    A box is frozen if it cannot move in EITHER axis, accounting for the fact
    that a neighbour box being frozen also blocks movement.

    Cycle handling: if we encounter a box already being evaluated (in_progress),
    we conservatively assume it is frozen — which is correct because mutually
    blocking boxes are indeed both immovable.
    """
    if box in memo:
        return memo[box]
    if box in in_progress:
        return True          # cycle → frozen

    r, c = box
    # A box sitting on its target is fine — skip it.
    if state.grid[r][c] == TARGET:
        memo[box] = False
        return False

    in_progress.add(box)

    # Horizontal: blocked on BOTH left and right?
    h_frozen = True
    for dc in (-1, 1):
        nc = c + dc
        if state.is_wall(r, nc):
            continue          # wall blocks this side
        nb = (r, nc)
        if nb in state.boxes and _is_frozen(nb, state, memo, in_progress):
            continue          # frozen box blocks this side
        h_frozen = False      # found a free or pushable side
        break

    # Vertical: blocked on BOTH above and below?
    v_frozen = True
    for dr in (-1, 1):
        nr = r + dr
        if state.is_wall(nr, c):
            continue
        nb = (nr, c)
        if nb in state.boxes and _is_frozen(nb, state, memo, in_progress):
            continue
        v_frozen = False
        break

    frozen = h_frozen and v_frozen
    in_progress.discard(box)
    memo[box] = frozen
    return frozen


def _freeze_deadlock(state: GameState) -> bool:
    """
    Returns True if any non-target box is frozen on both axes.
    Uses shared memo so each box is evaluated at most once per call.
    """
    memo: dict = {}
    in_progress: set = set()
    for box in state.boxes:
        if state.grid[box[0]][box[1]] == TARGET:
            continue
        if _is_frozen(box, state, memo, in_progress):
            return True
    return False


def _line_deadlock(state: GameState) -> bool:
    """
    Line deadlock: if a group of boxes is trapped along a wall in a single
    row or column, and there are fewer targets in that segment than boxes,
    at least one box can never reach a target — deadlock.

    A box is 'wall-pinned' vertically if it touches a horizontal wall
    (above or below), locking it to move only left/right within that row.
    Similarly for horizontal pinning.
    """
    rows = state.rows
    cols = state.cols
    grid = state.grid

    targets = frozenset((r, c) for r in range(rows) for c in range(cols)
                        if grid[r][c] == TARGET)

    # ── Horizontal line check (boxes pinned to top or bottom wall) ──
    for r in range(rows):
        for wall_side in (-1, 1):          # -1 = above, +1 = below
            nr = r + wall_side
            if nr < 0 or nr >= rows:
                continue
            # Collect boxes in this row that are pinned to this wall
            pinned = [c for (br, bc) in state.boxes
                      if br == r and state.is_wall(r + wall_side, bc)
                      for c in [bc]]
            if len(pinned) < 2:
                continue
            # Find contiguous segments between vertical walls in this row
            # Walls within the row act as boundaries
            seg_start = 0
            for c_end in range(cols + 1):
                is_boundary = (c_end == cols or state.is_wall(r, c_end))
                if is_boundary:
                    seg = [c for c in pinned if seg_start <= c < c_end]
                    if len(seg) >= 2:
                        tgts_in_seg = sum(1 for c in range(seg_start, c_end)
                                         if (r, c) in targets)
                        if tgts_in_seg < len(seg):
                            return True
                    seg_start = c_end + 1

    # ── Vertical line check (boxes pinned to left or right wall) ──
    for c in range(cols):
        for wall_side in (-1, 1):
            nc = c + wall_side
            if nc < 0 or nc >= cols:
                continue
            pinned = [r for (br, bc) in state.boxes
                      if bc == c and state.is_wall(br, c + wall_side)
                      for r in [br]]
            if len(pinned) < 2:
                continue
            seg_start = 0
            for r_end in range(rows + 1):
                is_boundary = (r_end == rows or state.is_wall(r_end, c))
                if is_boundary:
                    seg = [r for r in pinned if seg_start <= r < r_end]
                    if len(seg) >= 2:
                        tgts_in_seg = sum(1 for r in range(seg_start, r_end)
                                         if (r, c) in targets)
                        if tgts_in_seg < len(seg):
                            return True
                    seg_start = r_end + 1

    return False


def has_deadlock(state: GameState) -> bool:
    """
    Master deadlock check. Returns True if the state is provably unsolvable.

    Order: cheapest / broadest first.
      0. Dead zone  — O(1) lookup, catches wall-trapped boxes immediately
      1. Corner     — O(boxes)
      2. Freeze     — O(boxes) amortised, recursive
      3. Line       — O(rows*cols + boxes)
    """
    if _dead_zone_deadlock(state):
        return True
    if any(_corner_deadlock(state, b) for b in state.boxes):
        return True
    if _freeze_deadlock(state):
        return True
    if _line_deadlock(state):
        return True
    return False


def deadlock_type(state: GameState) -> str | None:
    """
    Returns a human-readable label for the first deadlock type found,
    or None if no deadlock. Used by the UI to display the reason.
    """
    if _dead_zone_deadlock(state):
        return "Dead zone — box is stuck against a wall"
    if any(_corner_deadlock(state, b) for b in state.boxes):
        return "Corner deadlock"
    if _freeze_deadlock(state):
        return "Freeze deadlock"
    if _line_deadlock(state):
        return "Line deadlock"
    return None


# ── Heuristic ─────────────────────────────────────────────────────────────────
def _get_targets(state: GameState) -> list[tuple[int,int]]:
    return [(r,c) for r in range(state.rows)
                  for c in range(state.cols)
                  if state.grid[r][c] == TARGET]

def heuristic(state: GameState, targets: list[tuple[int,int]]) -> int:
    total = 0
    for br, bc in state.boxes:
        if state.grid[br][bc] == TARGET:
            continue
        total += min(abs(br-tr)+abs(bc-tc) for tr,tc in targets)
    return total


# ── A* ────────────────────────────────────────────────────────────────────────
def solve(initial: GameState, max_nodes: int = 1_000_000) -> SolverResult:
    t0      = time.perf_counter()
    targets = _get_targets(initial)
    counter = 0
    heap    = [(heuristic(initial, targets), 0, counter, initial, [initial])]
    visited: set[tuple] = set()
    ne = 0;  ng = 1

    while heap:
        f, g, _, state, path = heapq.heappop(heap)
        key = (state.player, state.boxes)
        if key in visited:
            continue
        visited.add(key)
        ne += 1
        if ne > max_nodes:
            break

        if state.is_solved():
            rt     = (time.perf_counter() - t0) * 1000
            full   = _build_full_path(path)
            pushes = _extract_pushes(path)
            return SolverResult(True, full, pushes, ne, ng, rt, len(pushes))

        for dir_name in DIRECTIONS:
            ns = state.apply_move(dir_name)
            if ns is None or has_deadlock(ns):
                continue
            nk = (ns.player, ns.boxes)
            if nk in visited:
                continue
            ng += 1
            counter += 1
            new_g = g + 1
            heapq.heappush(heap, (new_g + heuristic(ns, targets),
                                  new_g, counter, ns, path + [ns]))

    rt = (time.perf_counter() - t0) * 1000
    return SolverResult(False, [], [], ne, ng, rt, 0)


# ── Path helpers ──────────────────────────────────────────────────────────────
def _build_full_path(states: list[GameState]) -> list[MoveStep]:
    steps = []
    for i in range(1, len(states)):
        prev = states[i-1];  curr = states[i]
        moved   = prev.boxes - curr.boxes
        arrived = curr.boxes - prev.boxes
        is_push = bool(moved)
        dr = curr.player[0] - prev.player[0]
        dc = curr.player[1] - prev.player[1]
        dir_name = next(d for d,(r,c) in DIRECTIONS.items() if r==dr and c==dc)
        steps.append(MoveStep(
            direction   = dir_name,
            is_push     = is_push,
            box_from    = next(iter(moved))   if moved   else None,
            box_to      = next(iter(arrived)) if arrived else None,
            state_after = curr,
        ))
    return steps


def _extract_pushes(states: list[GameState]) -> list[PushStep]:
    pushes = []
    for i in range(1, len(states)):
        prev = states[i-1];  curr = states[i]
        moved   = prev.boxes - curr.boxes
        arrived = curr.boxes - prev.boxes
        if not moved:
            continue
        box_from = next(iter(moved))
        box_to   = next(iter(arrived))
        dr = box_to[0] - box_from[0]
        dc = box_to[1] - box_from[1]
        dir_name = next(d for d,(r,c) in DIRECTIONS.items() if r==dr and c==dc)
        pushes.append(PushStep(dir_name, box_from, box_to,
                               prev.player, curr.player, curr))
    return pushes

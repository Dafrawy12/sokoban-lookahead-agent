"""
sokoban/game.py
Pure-logic Sokoban engine — no rendering dependencies.

Tile characters (ASCII / JSON levels):
  '#'  wall
  ' '  floor
  '.'  target
  '@'  player on floor
  '+'  player on target
  '$'  box on floor
  '*'  box on target
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import copy

# ── Tile constants ──────────────────────────────────────────────────────────
WALL    = '#'
FLOOR   = ' '
TARGET  = '.'
PLAYER  = '@'
PLAYER_ON_TARGET = '+'
BOX     = '$'
BOX_ON_TARGET    = '*'

DIRECTIONS = {
    'UP':    (-1,  0),
    'DOWN':  ( 1,  0),
    'LEFT':  ( 0, -1),
    'RIGHT': ( 0,  1),
}

# ── State ───────────────────────────────────────────────────────────────────
@dataclass
class GameState:
    """Immutable-friendly Sokoban state."""
    grid: list[list[str]]          # background tiles (walls, floors, targets)
    player: tuple[int, int]        # (row, col)
    boxes: frozenset[tuple[int, int]]
    rows: int
    cols: int

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_ascii(cls, lines: list[str]) -> "GameState":
        rows = len(lines)
        cols = max(len(l) for l in lines)
        grid: list[list[str]] = []
        player = (0, 0)
        boxes: set[tuple[int, int]] = set()

        for r, line in enumerate(lines):
            row: list[str] = []
            for c in range(cols):
                ch = line[c] if c < len(line) else ' '
                if ch == PLAYER:
                    player = (r, c)
                    row.append(FLOOR)
                elif ch == PLAYER_ON_TARGET:
                    player = (r, c)
                    row.append(TARGET)
                elif ch == BOX:
                    boxes.add((r, c))
                    row.append(FLOOR)
                elif ch == BOX_ON_TARGET:
                    boxes.add((r, c))
                    row.append(TARGET)
                else:
                    row.append(ch)
            grid.append(row)

        return cls(grid=grid, player=player,
                   boxes=frozenset(boxes), rows=rows, cols=cols)

    # ── queries ───────────────────────────────────────────────────────────
    def is_wall(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True
        return self.grid[r][c] == WALL

    def is_target(self, r: int, c: int) -> bool:
        return self.grid[r][c] == TARGET

    def is_solved(self) -> bool:
        return all(self.grid[r][c] == TARGET for r, c in self.boxes)

    def hashable(self) -> tuple:
        return (self.player, self.boxes)

    # ── mutation (returns new state or None if invalid) ────────────────────
    def apply_move(self, direction: str) -> Optional["GameState"]:
        dr, dc = DIRECTIONS[direction]
        pr, pc = self.player
        nr, nc = pr + dr, pc + dc          # new player pos

        if self.is_wall(nr, nc):
            return None

        new_boxes = set(self.boxes)

        if (nr, nc) in new_boxes:          # pushing a box
            br, bc = nr + dr, nc + dc      # box destination
            if self.is_wall(br, bc) or (br, bc) in new_boxes:
                return None                # blocked
            new_boxes.remove((nr, nc))
            new_boxes.add((br, bc))

        return GameState(
            grid=self.grid,
            player=(nr, nc),
            boxes=frozenset(new_boxes),
            rows=self.rows,
            cols=self.cols,
        )

    def render_ascii(self) -> str:
        """Return a printable string for debugging."""
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.player:
                    row.append(PLAYER_ON_TARGET if self.grid[r][c] == TARGET else PLAYER)
                elif (r, c) in self.boxes:
                    row.append(BOX_ON_TARGET if self.grid[r][c] == TARGET else BOX)
                else:
                    row.append(self.grid[r][c])
            lines.append(''.join(row))
        return '\n'.join(lines)


# ── Level loader ─────────────────────────────────────────────────────────────
import json, pathlib

def load_levels(path: str | pathlib.Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)

def level_to_state(level: dict) -> GameState:
    return GameState.from_ascii(level['grid'])

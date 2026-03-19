# Sokoban — A* Solver with Path Visualization
**Bachelor Thesis Project · SS26**

A polished Sokoban game implemented in Python/Pygame with a built-in A* solver,
animated path overlay, and rich solver statistics panel.

---

## Setup

```bash
pip install pygame
python main.py
```

Requires Python 3.10+ and Pygame 2.1+.

---

## Controls

| Key                | Action                              |
|--------------------|-------------------------------------|
| Arrow keys / WASD  | Move player                         |
| U                  | Undo last move                      |
| R                  | Reset level                         |
| N / P              | Next / Previous level               |
| L                  | Open level select screen            |
| Space              | Run A* solver (background thread)   |
| O                  | Toggle path overlay ON/OFF          |
| Enter              | Play / Pause solver animation       |
| ← / →              | Step backward / forward in solution |
| Home / End         | Jump to start / end of solution     |
| ↑ / ↓              | Increase / decrease playback speed  |

---

## Project Structure

```
sokoban/
├── main.py          # App entry point, event loop, UI state machine
├── game.py          # Pure game logic: GameState, move validation, win detection
├── solver.py        # A* push-based solver with deadlock detection
├── renderer.py      # All Pygame rendering, tile drawing, path overlay, panels
├── levels/
│   └── levels.json  # 8 sample levels (easy → medium) in JSON/ASCII format
└── requirements.txt
```

---

## Architecture Notes (for thesis)

### `game.py` — The World (Task 1)
- `GameState` stores the board grid (background only: walls/floor/targets) plus
  dynamic state (player position, box positions as a frozenset).
- `apply_move()` returns a **new** GameState (functional/immutable style) —
  essential for tree search where we explore many branches.
- `hashable()` produces a compact key `(player, boxes)` for visited-set lookups.

### `solver.py` — A* Search
- **Push-based search**: the search space is over box-push events, not individual
  player steps. This dramatically reduces the search space.
- **Zone merging**: two states that differ only in player position within the same
  connected region are treated as identical. The canonical key is the
  top-left reachable cell (`zone_key()`).
- **Heuristic**: sum of each box's distance to its nearest target (admissible,
  never overestimates → A* remains optimal).
- **Deadlock pruning**: corner deadlocks are detected and pruned immediately —
  a box stuck in a non-target corner can never be solved.
- **Stats reported**: solved/unsolved, pushes, nodes expanded, nodes generated,
  runtime in ms.

### `renderer.py` — Visualization
- Tile-based rendering with a dark industrial color palette.
- **Path overlay**: after solving, draws numbered badges and arrows for each push
  step. Steps are color-coded: done (dim), current (bright), future (faded).
- **Animated playback**: the app drives the overlay step-by-step at configurable
  speed (0.25×, 0.5×, 1×, 2×, 4×).

---

## Level Format

Levels are stored as JSON arrays of ASCII grids:

```
'#'  wall          ' '  floor
'.'  target         '@'  player
'$'  box            '*'  box on target
'+'  player on target
```

---

## Next Steps (Thesis Phase 2)

The solver and GameState are designed to be extended for **Tree of Thoughts**:

- `game.py` already provides `apply_move()` as the **physics engine** verifier —
  returns `None` for invalid moves, making it a natural pruning oracle.
- The A* `push_sequence` output can serve as a **ground truth** baseline to
  compare against the ToT agent's lookahead path.
- Phase 2 will add: ToT agent, LLM candidate generation, move verifier, and
  the LLM explainability commentary layer.

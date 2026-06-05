"""
sokoban/pure_llm_agent.py — Pure LLM agent (Ollama/Mistral)

Shows the full board once, gets back a complete move sequence.
No verification, no filtering. Physically impossible moves silently skipped.
"""
from __future__ import annotations
import re, threading
from typing import Callable
from game import GameState, TARGET, DIRECTIONS

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen2.5:7b"
MAX_STEPS  = 300


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
    board_str  = "\n".join(rows)
    targets    = [(r,c) for r in range(state.rows) for c in range(state.cols)
                  if state.grid[r][c] == TARGET]
    off_target = [b for b in state.boxes if state.grid[b[0]][b[1]] != TARGET]
    free_tgts  = [t for t in targets if t not in state.boxes]
    return (f"{board_str}\n\n"
            f"Symbols: @ player  $ box  . target  * solved  # wall\n"
            f"Player: row={state.player[0]}, col={state.player[1]}\n"
            f"Boxes not on target: {off_target}\n"
            f"Free targets: {free_tgts}")


def _build_prompt(state: GameState) -> str:
    return f"""You are solving a Sokoban puzzle. Give the COMPLETE move sequence to solve it.

RULES:
- Move player (@) using: UP, DOWN, LEFT, RIGHT
- To push a box ($): walk the player INTO the box — it moves one step in that direction
- You CANNOT push a box into a wall or another box
- You CANNOT pull boxes — only push
- Goal: push ALL boxes ($) onto targets (.) by walking into them

{_board_to_text(state)}

IMPORTANT: The player must physically walk next to a box and then walk into it to push it.
To push box at (row,col) RIGHT: player must be at (row, col-1) then move RIGHT.
To push box at (row,col) DOWN: player must be at (row-1, col) then move DOWN.

Think about where the player needs to go to push each box onto its target.
Reply with ONLY a comma-separated move list. Example:
UP, LEFT, LEFT, DOWN, RIGHT

Full solution:"""


def _call_ollama(prompt: str) -> str:
    import urllib.request, json
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.4, "num_predict": 300}
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("response", "").strip()


def _parse_moves(text: str) -> list[str]:
    moves = []
    for word in re.split(r'[\s,.|:\-\n\(\)]+', text.upper()):
        if word.strip() in DIRECTIONS:
            moves.append(word.strip())
    return moves


class PureLLMAgent:
    def __init__(self, on_solution: Callable, on_status: Callable, on_error: Callable):
        self.on_solution = on_solution
        self.on_status   = on_status
        self.on_error    = on_error
        self._stop_evt   = threading.Event()
        self._thread     = None
        self.running     = False
        self.steps_taken = 0
        self.compute_ms  = 0.0
        self.rejections  = 0
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        except Exception as e:
            raise RuntimeError(f"Ollama not running: {e}")

    def start(self, initial_state: GameState):
        if self.running: return
        self._stop_evt.clear()
        self.running=True; self.steps_taken=0; self.compute_ms=0.0; self.rejections=0
        self._thread = threading.Thread(target=self._solve, args=(initial_state,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set(); self.running = False

    def _solve(self, initial_state: GameState):
        import time as _t
        try:
            self.on_status("Asking LLM for full solution...")
            t0 = _t.perf_counter()
            try:
                raw   = _call_ollama(_build_prompt(initial_state))
                moves = _parse_moves(raw)
                self.compute_ms = (_t.perf_counter()-t0)*1000
                self.on_status(f"Got {len(moves)} moves — applying...")
                self.on_status(f"[LLM] {raw[:200]}")
            except Exception as e:
                self.on_error(f"Ollama error: {e}"); return

            if not moves:
                self.on_error("LLM returned no moves"); return

            state = initial_state
            path  = []
            for direction in moves:
                if self._stop_evt.is_set(): break
                ns = state.apply_move(direction)
                if ns is None:
                    self.rejections += 1; continue
                path.append((direction, [], "pure llm"))
                state = ns; self.steps_taken += 1
                if state.is_solved(): break

            if state.is_solved():
                self.on_status(f"Solved in {self.steps_taken} steps ({self.compute_ms:.0f}ms)")
                self.on_solution(path)
            else:
                boxes_left = sum(1 for b in state.boxes if state.grid[b[0]][b[1]] != TARGET)
                self.on_status(f"Not solved — {self.steps_taken} moves, {boxes_left} box(es) left")
                if path: self.on_solution(path)
                else: self.on_error("No applicable moves")
        except Exception as e:
            self.on_error(f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.running = False

"""
sokoban/main.py

Two-arena layout:
  LEFT  — Pure LLM agent  (G to start)
  RIGHT — Hybrid LLM+ToT  (T to start)

B key — Results page: runs A*, Pure LLM, Hybrid on all 10 levels headlessly
        and shows a live comparison table.
M key — Manual play mode on left arena (Truth Scanner)
"""

from __future__ import annotations
import sys, time, pathlib, threading
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
import pygame

from game import GameState, load_levels, level_to_state, DIRECTIONS
from solver import solve, SolverResult, MoveStep, deadlock_type
from renderer import Renderer, TILE, PANEL_W, C
from llm_tot_agent import LLMToTAgent
from pure_llm_agent import PureLLMAgent

LEVELS_PATH = pathlib.Path(__file__).parent / 'levels' / 'levels_all.json'

FPS        = 60
SPEEDS     = [0.25, 0.5, 1.0, 2.0, 4.0]
TOT_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MOVE_DUR   = 0.12
PUSH_DUR   = 0.18
PLAY_DUR   = 0.45
ARENA_GAP  = 32
ARENA_PAD  = 16


def ease_out_cubic(t):
    return 1.0 - (1.0 - t) ** 3


# ── Animation ─────────────────────────────────────────────────────────────────
@dataclass
class Anim:
    player_from: tuple
    player_to:   tuple
    box_from:    Optional[tuple]
    box_to:      Optional[tuple]
    t:           float
    duration:    float
    prev_state:  GameState
    next_state:  GameState

    @property
    def done(self): return self.t >= 1.0
    def advance(self, dt): self.t = min(1.0, self.t + dt / self.duration)
    def alpha(self): return ease_out_cubic(self.t)


# ── Arena ─────────────────────────────────────────────────────────────────────
class Arena:
    def __init__(self, label, initial, level):
        self.label   = label
        self.level   = level
        self.state   = initial
        self.anim    = None
        self.history = []
        self.solver_result = None
        self.solving  = False
        self.playing  = False
        self.playback_step = -1
        self.show_overlay  = False
        self.status        = ''
        self.log: list[str] = []
        self.log_scroll: int = 0

    def reset(self, level):
        self.level   = level
        self.state   = level_to_state(level)
        self.anim    = None
        self.history = []
        self.solver_result = None
        self.solving  = False
        self.playing  = False
        self.playback_step = -1
        self.status   = ''
        # NOTE: log and log_scroll are NOT reset here — preserved across replays

    def reset_full(self, level):
        """Full reset including log — used on level change only."""
        self.reset(level)
        self.log        = []
        self.log_scroll = 0

    def add_log(self, msg: str):
        self.log.append(msg)
        if len(self.log) > 500:
            self.log = self.log[-500:]
        # Only auto-scroll to bottom if user is already at bottom (scroll == 0)
        if self.log_scroll == 0:
            pass   # already at bottom, nothing to do
        # If user has scrolled up, don't jump them back down

    def start_anim(self, prev, nxt, box_from=None, box_to=None, dur=MOVE_DUR):
        self.anim = Anim(player_from=prev.player, player_to=nxt.player,
                         box_from=box_from, box_to=box_to,
                         t=0.0, duration=dur, prev_state=prev, next_state=nxt)

    def snap_anim(self):
        if self.anim and not self.anim.done:
            self.state = self.anim.next_state
        self.anim = None

    def advance_anim(self, dt):
        if self.anim:
            self.anim.advance(dt)
            if self.anim.done:
                self.anim = None

    def do_move(self, direction, speed=1.0):
        self.snap_anim()
        prev = self.state
        new_state = prev.apply_move(direction)
        if not new_state: return False
        self.history.append(prev)
        dr, dc   = DIRECTIONS[direction]
        nr, nc   = prev.player[0]+dr, prev.player[1]+dc
        pushed   = (nr,nc) in prev.boxes
        box_from = (nr,nc)       if pushed else None
        box_to   = (nr+dr,nc+dc) if pushed else None
        dur      = (PUSH_DUR if pushed else MOVE_DUR) / speed
        self.state = new_state
        self.start_anim(prev, new_state, box_from, box_to, dur)
        return True


# ── Benchmark worker ──────────────────────────────────────────────────────────
class BenchmarkRunner:
    """Runs A*, Pure LLM, Hybrid on all levels headlessly. Thread-safe results."""

    def __init__(self, levels, on_update, app=None):
        self.levels    = levels
        self.on_update = on_update
        self._app      = app
        self.results   = {}
        self.running   = False
        self._thread   = None
        self._stop_evt = threading.Event()

    def start(self):
        if self.running: return
        self.running = True
        self.results = {}
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self.running = False

    def _run(self):
        import time as _t
        agents = ["astar", "pure_llm", "hybrid"]
        for level in self.levels:
            for agent in agents:
                if self._stop_evt.is_set(): return
                key = (agent, level["id"])
                self.results[key] = {"status": "running..."}
                self.on_update()
                try:
                    if agent == "astar":
                        res = self._run_astar(level)
                    elif agent == "pure_llm":
                        res = self._run_agent(PureLLMAgent, level)
                    else:
                        res = self._run_agent(LLMToTAgent, level)
                    self.results[key] = res
                except Exception as e:
                    self.results[key] = {"status": f"error: {e}"}
                self.on_update()
        self.running = False
        self.on_update()

    def _run_astar(self, level) -> dict:
        import time as _t
        state = level_to_state(level)
        t0    = _t.perf_counter()
        res   = solve(state)
        ms    = (_t.perf_counter()-t0)*1000
        return {
            "status":     "solved" if res.solved else "unsolved",
            "steps":      len(res.full_path) if res.solved else "-",
            "pushes":     res.num_pushes     if res.solved else "-",
            "nodes":      res.nodes_expanded,
            "compute_ms": round(res.runtime_ms, 1),
        }

    def _run_agent(self, AgentClass, level) -> dict:
        import time as _t
        done_evt = threading.Event()
        result   = {}

        def on_solution(path):
            result["path"] = path
            done_evt.set()
        def on_status(msg): pass
        def on_error(msg):
            result["error"] = msg
            done_evt.set()

        agent = AgentClass(on_solution, on_status, on_error)
        t0    = _t.perf_counter()
        agent.start(level_to_state(level))
        done_evt.wait(timeout=350)  # 100 steps × 2.7s + buffer = ~350s max
        wall_ms = (_t.perf_counter()-t0)*1000
        agent.stop()

        # If neither solution nor error arrived → timeout
        if "path" not in result and "error" not in result:
            result["error"] = "timeout"

        # Verify the solution actually solves the level
        if "path" in result:
            state = level_to_state(level)
            for direction, *_ in result["path"]:
                ns = state.apply_move(direction)
                if ns is not None:
                    state = ns
            if not state.is_solved():
                result["error"] = "path did not solve level"
                del result["path"]

        solved = "path" in result
        path   = result.get("path", [])
        pushes = 0
        if solved:
            state = level_to_state(level)
            for direction, _, _ in path:
                ns = state.apply_move(direction)
                if ns and ns.boxes != state.boxes: pushes += 1
                if ns: state = ns

        return {
            "status":           "solved" if solved else "unsolved",
            "steps":            agent.steps_taken if solved else "-",
            "pushes":           pushes if solved else "-",
            "compute_ms":       round(agent.compute_ms, 1),
            "backtracks":       getattr(agent, "backtracks", "-"),
            "nodes":            getattr(agent, "nodes", "-"),
            "illegal_rejected": getattr(agent, "illegal_rejected", "-"),
            "deadlock_rejected":getattr(agent, "deadlock_rejected", "-"),
        }


# ── App ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Sokoban  --  LLM Agents')

        self.levels     = load_levels(LEVELS_PATH)
        self.level_idx  = 0
        self.mode       = 'game'   # 'game' | 'select' | 'results'
        self.select_idx = 0
        self.speed_idx  = 2
        self.tot_speed_idx = 2

        lvl = self.levels[0]
        self.left  = Arena('Pure LLM', level_to_state(lvl), lvl)
        self.right = Arena('Hybrid',   level_to_state(lvl), lvl)

        self._init_screen()

        icon = pygame.Surface((32,32))
        icon.fill(C['box'])
        pygame.draw.rect(icon, C['box_solved'], (8,8,16,16))
        pygame.display.set_icon(icon)

        self.clock    = pygame.time.Clock()
        self.renderer = Renderer(self.screen)

        # ── Pure LLM agent ────────────────────────────────────────────────
        try:
            self.pure_agent = PureLLMAgent(
                on_solution=self._on_pure_solution,
                on_status=self._on_pure_status,
                on_error=self._on_pure_error,
            )
        except Exception as e:
            self.pure_agent = None
            print(f"Pure LLM unavailable: {e}")

        # ── Hybrid agent ──────────────────────────────────────────────────
        try:
            self.hybrid_agent = LLMToTAgent(
                on_solution=self._on_hybrid_solution,
                on_status=self._on_hybrid_status,
                on_error=self._on_hybrid_error,
                on_step=self._on_hybrid_step,
            )
        except Exception as e:
            self.hybrid_agent = None
            print(f"Hybrid unavailable: {e}")

        # ── Shared solution playback ──────────────────────────────────────
        self.left_solution:  list = []
        self.left_sol_idx:   int  = -1
        self.left_playing:   bool = False

        self.right_solution: list = []
        self.right_sol_idx:  int  = -1
        self.right_playing:  bool = False

        # ── Benchmark ─────────────────────────────────────────────────────
        self.bench = BenchmarkRunner(self.levels, self._on_bench_update, app=self)
        self._bench_dirty = False
        self.bench_run_astar    = True   # toggle A* in benchmark
        self.bench_run_pure     = True   # toggle Pure LLM in benchmark

        # ── Manual mode ───────────────────────────────────────────────────

        # ── Replay mode (after animation finishes) ────────────────────────
        # When playing=False and solution exists, user can step through manually
        self.left_replay  = False   # True once animation done
        self.right_replay = False

        # ── Mouse scroll drag ─────────────────────────────────────────────
        self._drag_arena       = None
        self._drag_start_y     = 0
        self._drag_start_scroll = 0
        self._log_rects        = {}

        # ── Hybrid live animation queue ───────────────────────────────────
        import queue as _q
        self._hybrid_queue = _q.Queue()   # receives directions as agent solves

    # ── Agent callbacks ───────────────────────────────────────────────────

    def _on_pure_solution(self, path):
        self.left_solution = path
        self.left_sol_idx  = -1
        self.left_playing  = True
        self.left_replay   = False
        msg = f'Got {len(path)} moves -- animating...'
        self.left.status = msg
        self.left.add_log(msg)

    def _on_pure_status(self, msg):
        self.left.status = msg
        self.left.add_log(msg)

    def _on_pure_error(self, msg):
        self.left.status  = f'Error: {msg}'
        self.left_playing = False
        self.left.add_log(f'ERROR: {msg}')

    def _on_hybrid_step(self, signal):
        """Called by hybrid agent — queue for live animation."""
        self._hybrid_queue.put(signal)

    def _on_hybrid_solution(self, path):
        # Store full path for replay
        self.right_solution = path
        self.right_sol_idx  = -1
        self.right_replay   = True   # go straight to replay mode
        self.right_playing  = False
        # Reset board to initial state so replay starts clean
        self.right.reset(self.right.level)
        ms  = self.hybrid_agent.compute_ms if self.hybrid_agent else 0
        msg = (f'Solved -- {len(path)} steps  {ms:.0f}ms  '
               f'| <- -> to replay')
        self.right.status = msg
        self.right.add_log(msg)

    def _on_hybrid_status(self, msg):
        self.right.status = msg
        self.right.add_log(msg)

    def _on_hybrid_error(self, msg):
        self.right.status  = f'Error: {msg}'
        self.right_playing = False
        self.right.add_log(f'ERROR: {msg}')

    def _on_bench_update(self):
        self._bench_dirty = True

    # ── Layout ────────────────────────────────────────────────────────────

    def _tile_size(self):
        """Calculate tile size based on the largest level so window never resizes."""
        import pygame, renderer as _r
        info  = pygame.display.Info()
        max_w = info.current_w - 40
        max_h = info.current_h - 80
        # Find largest board across all levels
        max_cols = max(level_to_state(l).cols for l in self.levels)
        max_rows = max(level_to_state(l).rows for l in self.levels)
        tile = TILE
        while tile > 20:
            win_w = ARENA_PAD + max_cols*tile*2 + ARENA_GAP + ARENA_PAD + PANEL_W
            win_h = max_rows * tile + 250
            if win_w <= max_w and win_h <= max_h:
                break
            tile -= 2
        return tile

    def _board_size(self):
        state = level_to_state(self.levels[self.level_idx])
        tile  = self._tile_size()
        return state.cols * tile, state.rows * tile

    def _init_screen(self):
        import renderer as _r
        tile = self._tile_size()
        _r.TILE = tile
        if hasattr(self, 'renderer'):
            self.renderer._tile_cache.clear()
        # Window size based on LARGEST level — never changes
        max_cols = max(level_to_state(l).cols for l in self.levels)
        max_rows = max(level_to_state(l).rows for l in self.levels)
        win_w = ARENA_PAD + max_cols*tile*2 + ARENA_GAP + ARENA_PAD + PANEL_W
        win_h = max(800, max_rows * tile + 250)
        if not hasattr(self, 'screen'):
            self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        self._calc_offsets()

    def _calc_offsets(self):
        sw, sh   = self.screen.get_size()
        bw, bh   = self._board_size()
        boards_w = bw*2 + ARENA_GAP
        start_x  = max(ARENA_PAD, (sw - PANEL_W - boards_w)//2)
        oy       = max(50, min(60, (sh - bh - 200)//2))  # push board up, leave 200px for log
        self.left_offset  = (start_x, oy)
        self.right_offset = (start_x + bw + ARENA_GAP, oy)

    # ── Level init ────────────────────────────────────────────────────────

    def _init_level(self):
        if self.pure_agent:   self.pure_agent.stop()
        if self.hybrid_agent: self.hybrid_agent.stop()
        lvl = self.levels[self.level_idx]
        self.left.reset_full(lvl)
        self.right.reset_full(lvl)
        self.left_solution  = []; self.left_sol_idx  = -1; self.left_playing  = False
        self.right_solution = []; self.right_sol_idx = -1; self.right_playing = False
        self.left_replay  = False
        self.right_replay = False
        while not self._hybrid_queue.empty():
            try: self._hybrid_queue.get_nowait()
            except: pass
        self.manual_mode = False;
        self._init_screen()

    # ── Agent toggles ─────────────────────────────────────────────────────

    def _toggle_pure(self):
        if not self.pure_agent:
            self.left.status = 'Ollama not running'; return
        if self.pure_agent.running:
            self.pure_agent.stop()
            self.left.status = 'Stopped'
        else:
            self.left.reset(self.left.level)
            self.left_solution = []; self.left_sol_idx = -1; self.left_playing = False
            self.left.status = 'Pure LLM solving...'
            print("[G] Starting Pure LLM agent...")
            self.pure_agent.start(level_to_state(self.left.level))

    def _toggle_hybrid(self):
        if not self.hybrid_agent:
            self.right.status = 'Ollama not running'; return
        if self.hybrid_agent.running:
            self.hybrid_agent.stop()
            self.right.status = 'Stopped'
        else:
            self.right.reset(self.right.level)
            self.right_solution = []; self.right_sol_idx = -1; self.right_playing = False
            self.right.status = 'Hybrid LLM+ToT solving...'
            print("[T] Starting Hybrid agent...")
            self.hybrid_agent.start(level_to_state(self.right.level))

    # ── Truth Scanner ─────────────────────────────────────────────────────

    def _replay_step_arena(self, arena, solution, idx_attr, replay_attr, direction):
        """Step forward or back through a solution for one arena."""
        if not getattr(self, replay_attr) or not solution:
            return
        if arena.anim is not None:
            return
        speed   = TOT_SPEEDS[self.tot_speed_idx]
        cur_idx = getattr(self, idx_attr)
        new_idx = cur_idx + direction

        # Clamp: -1 = initial state, 0..len-1 = solution steps
        if new_idx < -1 or new_idx >= len(solution):
            return

        # Reset board to initial then replay up to new_idx
        arena.reset(arena.level)
        for i in range(new_idx + 1):
            arena.snap_anim()
            arena.do_move(solution[i][0], speed=999)
        setattr(self, idx_attr, new_idx)

        if new_idx == -1:
            arena.status = 'Replay: initial state'
        else:
            arena.status = f'Replay {new_idx+1}/{len(solution)}'

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.VIDEORESIZE:
                self._calc_offsets()
            if self.mode == 'select':
                self._handle_select(ev)
            elif self.mode == 'results':
                self._handle_results(ev)
            else:
                self._handle_game(ev)
                self._handle_mouse(ev)

    def _handle_game(self, ev):
        if ev.type != pygame.KEYDOWN: return
        k = ev.key

        if k == pygame.K_l:
            self.mode = 'select'; self.select_idx = self.level_idx; return
        if k == pygame.K_r:
            self._init_level(); return
        if k == pygame.K_n:
            self.level_idx = (self.level_idx+1) % len(self.levels)
            self._init_level(); return
        if k == pygame.K_p:
            self.level_idx = (self.level_idx-1) % len(self.levels)
            self._init_level(); return
        if k == pygame.K_g:
            self._toggle_pure(); return
        if k == pygame.K_t:
            self._toggle_hybrid(); return
        if k == pygame.K_b:
            self.mode = 'results'
            if not self.bench.running:
                self.bench.results = {}
                self.bench.start()
            return
        if k == pygame.K_v:
            self.tot_speed_idx = min(self.tot_speed_idx+1, len(TOT_SPEEDS)-1); return
        if k == pygame.K_c:
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_CTRL:
                # Ctrl+C — copy right log to clipboard
                text = '\n'.join(self.right.log)
                pygame.scrap.init()
                pygame.scrap.put(pygame.SCRAP_TEXT, text.encode())
                self.right.status = 'Log copied to clipboard!'
                return
            self.tot_speed_idx = max(self.tot_speed_idx-1, 0); return

        # Left board controls: A/D = replay, W/S = scroll log
        if k == pygame.K_w:
            self.left.log_scroll = min(self.left.log_scroll+1, max(0,len(self.left.log)-1))
        elif k == pygame.K_s:
            self.left.log_scroll = max(self.left.log_scroll-1, 0)
        elif k == pygame.K_a:
            self._replay_step_arena(self.left, self.left_solution,
                                    'left_sol_idx', 'left_replay', -1)
        elif k == pygame.K_d:
            self._replay_step_arena(self.left, self.left_solution,
                                    'left_sol_idx', 'left_replay', +1)

        # Right board controls: arrow UP/DOWN = scroll, LEFT/RIGHT = replay
        elif k == pygame.K_UP:
            self.right.log_scroll = min(self.right.log_scroll+1, max(0,len(self.right.log)-1))
        elif k == pygame.K_DOWN:
            self.right.log_scroll = max(self.right.log_scroll-1, 0)
        elif k == pygame.K_LEFT:
            self._replay_step_arena(self.right, self.right_solution,
                                    'right_sol_idx', 'right_replay', -1)
        elif k == pygame.K_RIGHT:
            self._replay_step_arena(self.right, self.right_solution,
                                    'right_sol_idx', 'right_replay', +1)

    def _handle_select(self, ev):
        if ev.type != pygame.KEYDOWN: return
        k, n = ev.key, len(self.levels)
        if   k == pygame.K_RIGHT: self.select_idx = (self.select_idx+1)%n
        elif k == pygame.K_LEFT:  self.select_idx = (self.select_idx-1)%n
        elif k == pygame.K_DOWN:  self.select_idx = (self.select_idx+4)%n
        elif k == pygame.K_UP:    self.select_idx = (self.select_idx-4)%n
        elif k in (pygame.K_RETURN, pygame.K_SPACE):
            self.level_idx = self.select_idx
            self._init_level(); self.mode = 'game'
        elif k == pygame.K_ESCAPE:
            self.mode = 'game'

    def _handle_results(self, ev):
        if ev.type != pygame.KEYDOWN: return
        if ev.key in (pygame.K_ESCAPE, pygame.K_b):
            self.mode = 'game'
        elif ev.key == pygame.K_r:
            self.bench.stop()
            time.sleep(0.2)
            self.bench.results = {}
            self.bench.start()
        elif ev.key == pygame.K_1:
            self.bench_run_astar = not self.bench_run_astar
        elif ev.key == pygame.K_2:
            self.bench_run_pure = not self.bench_run_pure

    def _handle_mouse(self, ev):
        """Mouse wheel scroll + click-drag scrollbar for log boxes."""
        # Mouse wheel — scroll whichever log the cursor is over
        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            for arena, rect in self._log_rects.items():
                if rect.collidepoint(mx, my):
                    arena.log_scroll = max(0, arena.log_scroll - ev.y)
                    # clamp handled in _draw_log

        # Mouse button down — check if on a scrollbar
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            for key, (arena, rect) in [('left', (self.left, self._log_rects.get(self.left))),
                                        ('right', (self.right, self._log_rects.get(self.right)))]:
                if rect is None: continue
                # Scrollbar is the rightmost 7px of the log rect
                sb_rect = pygame.Rect(rect.x + rect.w - 7, rect.y, 7, rect.h)
                if sb_rect.collidepoint(mx, my):
                    self._drag_arena        = key
                    self._drag_start_y      = my
                    self._drag_start_scroll = arena.log_scroll

        # Mouse move — drag scrollbar
        elif ev.type == pygame.MOUSEMOTION and self._drag_arena:
            _, my = ev.pos
            dy    = my - self._drag_start_y
            arena = self.left if self._drag_arena == 'left' else self.right
            rect  = self._log_rects.get(arena)
            if rect and rect.h > 0:
                # Work out how many wrapped lines exist
                total = self._get_wrapped_line_count(arena, rect.w)
                # Map pixel delta to scroll units
                scroll_range = max(1, total - max(1, rect.h // 16))
                delta = int(dy / rect.h * scroll_range)
                arena.log_scroll = max(0, self._drag_start_scroll - delta)

        # Mouse button up — end drag
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._drag_arena = None

    # ── Update ────────────────────────────────────────────────────────────

    def update(self, dt):
        self.left.advance_anim(dt)
        self.right.advance_anim(dt)

        tot_speed = TOT_SPEEDS[self.tot_speed_idx]

        # Left solution playback
        if self.left_playing and self.left_solution and self.left.anim is None:
            self.left_sol_idx += 1
            if self.left_sol_idx < len(self.left_solution):
                direction = self.left_solution[self.left_sol_idx][0]
                self.left.do_move(direction, speed=tot_speed)
                self.left.status = (f'Step {self.left_sol_idx+1}/'
                                    f'{len(self.left_solution)}')
            else:
                self.left_playing = False
                self.left_replay  = True
                self.left_sol_idx = -1   # reset so first D press goes to step 0
                ms  = self.pure_agent.compute_ms if self.pure_agent else 0
                sol = len(self.left_solution)
                self.left.status = (f'Done -- {sol} steps  {ms:.0f}ms  '
                                    f'| A/D to replay')
                self.left.add_log(f'Done. Press D to step forward, A to step back.')
                # Reset board to initial state for clean replay
                self.left.reset(self.left.level)

        # Right arena — live hybrid animation (drains queue as agent solves)
        if self.right.anim is None and not self._hybrid_queue.empty():
            try:
                signal = self._hybrid_queue.get_nowait()
                if isinstance(signal, tuple) and signal[0] == "PATH":
                    moves = signal[1]
                    self.right.reset(self.right.level)
                    for m in moves:
                        self.right.snap_anim()
                        self.right.do_move(m, speed=999)
                    self.right.status = f'Exploring depth {len(moves)}...'
                elif isinstance(signal, tuple) and signal[0] == "BACKTRACK":
                    # Show backtrack at normal animation speed
                    moves = signal[1]
                    self.right.reset(self.right.level)
                    for m in moves:
                        self.right.snap_anim()
                        self.right.do_move(m, speed=999)
                    self.right.status = f'Backtracking... depth {len(moves)}'
                elif isinstance(signal, str):
                    self.right.do_move(signal, speed=tot_speed)
            except Exception:
                pass

        # Right replay playback (after solution received)
        if self.right_playing and self.right_solution and self.right.anim is None:
            self.right_sol_idx += 1
            if self.right_sol_idx < len(self.right_solution):
                direction = self.right_solution[self.right_sol_idx][0]
                self.right.do_move(direction, speed=tot_speed)
                self.right.status = (f'Replay {self.right_sol_idx+1}/'
                                     f'{len(self.right_solution)}')
            else:
                self.right_playing = False
                self.right_replay  = True
                self.right_sol_idx = -1
                ms  = self.hybrid_agent.compute_ms if self.hybrid_agent else 0
                sol = len(self.right_solution)
                self.right.status = (f'Done -- {sol} steps  {ms:.0f}ms  '
                                     f'| <- -> to replay')
                self.right.add_log('Done. Press -> to step forward, <- to step back.')
                self.right.reset(self.right.level)

    # ── Draw ──────────────────────────────────────────────────────────────

    def draw(self):
        if self.mode == 'results':
            self._draw_results()
            return

        self._log_rects = {}   # reset each frame
        sw, sh = self.screen.get_size()
        self.screen.fill(C['bg'])
        self._calc_offsets()
        bw, bh   = self._board_size()
        lox, loy = self.left_offset
        rox, roy = self.right_offset

        # Labels
        left_lbl = 'Pure LLM Agent'
        left_col = C['accent']
        self.screen.blit(
            self.renderer.font_title.render(left_lbl, True, left_col), (lox, 10))
        self.screen.blit(
            self.renderer.font_title.render('Hybrid LLM + ToT', True, C['accent2']),
            (rox, 10))

        # Status lines below labels
        self.screen.blit(
            self.renderer.font_small.render(self.left.status[:60], True, C['text_dim']),
            (lox, 27))
        self.screen.blit(
            self.renderer.font_small.render(self.right.status[:60], True, C['text_dim']),
            (rox, 27))

        # Boards
        self.renderer.draw_board(
            state=self.left.state, offset=self.left_offset,
            overlay_steps=None, current_step=-1,
            show_overlay=False, anim=self.left.anim)
        self.renderer.draw_board(
            state=self.right.state, offset=self.right_offset,
            overlay_steps=None, current_step=-1,
            show_overlay=False, anim=self.right.anim)

        # Divider
        div_x = rox - ARENA_GAP//2
        pygame.draw.line(self.screen, C['wall_hi'], (div_x,0), (div_x,sh), 1)

        # Win overlays
        self._draw_win_overlay(self.left.state,  self.left_offset,  bw, bh, C['accent'])
        self._draw_win_overlay(self.right.state, self.right_offset, bw, bh, C['accent2'])

        # Logs under boards
        log_y = loy + bh + 8
        log_h = sh - log_y - 8
        self._draw_log(self.left,  lox,  log_y, bw, log_h)
        self._draw_log(self.right, rox,  log_y, bw, log_h)

        # Panel
        panel_rect = pygame.Rect(sw-PANEL_W, 0, PANEL_W, sh)
        pygame.draw.line(self.screen, C['wall_hi'], (sw-PANEL_W,0), (sw-PANEL_W,sh), 1)
        self._draw_panel(panel_rect)

        if self.mode == 'select':
            self.renderer.draw_level_select(self.levels, self.select_idx, sw, sh)

        pygame.display.flip()

    def _draw_win_overlay(self, state, offset, bw, bh, colour):
        if not state.is_solved(): return
        ox, oy = offset
        s = pygame.Surface((bw, 40), pygame.SRCALPHA)
        s.fill((*colour, 180))
        self.screen.blit(s, (ox, oy + bh//2 - 20))
        txt = self.renderer.font_body.render('SOLVED!', True, (255,255,255))
        self.screen.blit(txt, txt.get_rect(center=(ox+bw//2, oy+bh//2)))

    def _get_wrapped_line_count(self, arena: Arena, box_w: int) -> int:
        """Count total wrapped display lines for an arena's log."""
        font  = self.renderer.font_small
        max_w = box_w - 12 - 6
        count = 0
        for entry in arena.log:
            words = entry.split(' ')
            cur   = ''
            for word in words:
                test = (cur+' '+word).strip()
                if font.size(test)[0] <= max_w:
                    cur = test
                else:
                    if cur: count += 1
                    cur = word
            count += 1
        return max(count, 1)

    def _draw_log(self, arena: Arena, x: int, y: int, w: int, h: int):
        """Draw scrollable log box under a board."""
        if h < 20:
            return
        surf = self.screen

        # Store rect for mouse hit testing
        self._log_rects[arena] = pygame.Rect(x, y, w, h)

        # Background
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((20, 20, 30, 200))
        surf.blit(bg, (x, y))
        pygame.draw.rect(surf, C['wall_hi'], (x, y, w, h), 1)

        font   = self.renderer.font_small
        line_h = font.get_height() + 2
        pad    = 6
        max_w  = w - pad*2 - 6   # leave room for scrollbar

        if not arena.log:
            surf.blit(font.render('No log yet...', True, C['text_dim']), (x+pad, y+pad))
            return

        # Word-wrap all log entries into display lines
        def wrap(text):
            words = text.split(' ')
            lines, cur = [], ''
            for word in words:
                test = (cur+' '+word).strip()
                if font.size(test)[0] <= max_w:
                    cur = test
                else:
                    if cur: lines.append(cur)
                    cur = word
            if cur: lines.append(cur)
            return lines or ['']

        def line_col(text):
            if 'ERROR' in text or 'error' in text: return C['error']
            if 'Solved' in text or 'solved' in text: return C['success']
            if 'Partial' in text or 'Not solved' in text: return (255,165,0)
            if '[LLM]' in text or '[Hybrid]' in text: return C['accent2']
            return C['text_dim']

        # Build full wrapped line list
        all_lines = []
        for entry in arena.log:
            col   = line_col(entry)
            for wl in wrap(entry):
                all_lines.append((wl, col))

        max_lines = max(1, (h - pad*2) // line_h)
        total     = len(all_lines)

        # scroll=0 means bottom, scroll=N means N lines up from bottom
        # Clamp scroll so you can't scroll past the top
        arena.log_scroll = min(arena.log_scroll, max(0, total - max_lines))

        end_idx   = total - arena.log_scroll
        end_idx   = max(end_idx, min(max_lines, total))
        start_idx = max(end_idx - max_lines, 0)
        visible   = all_lines[start_idx:end_idx]

        for i, (line, col) in enumerate(visible):
            surf.blit(font.render(line, True, col), (x+pad, y+pad + i*line_h))

        # Scroll bar + hint
        if total > max_lines:
            scroll_pct = 1.0 - (arena.log_scroll / max(total - max_lines, 1))
            bar_h = max(14, int(h * max_lines / total))
            bar_y = y + int(scroll_pct * (h - bar_h))
            # Track
            pygame.draw.rect(surf, (40,40,60), (x+w-7, y, 6, h))
            # Thumb
            pygame.draw.rect(surf, C['accent'], (x+w-7, bar_y, 6, bar_h), border_radius=3)
            # Hint at bottom right
            hint = font.render('W/S scroll', True, C['text_dim'])
            surf.blit(hint, (x+w-hint.get_width()-10, y+h-line_h-2))

    def _draw_panel(self, rect):
        from renderer import draw_rounded_rect
        surf = self.screen
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        draw_rounded_rect(surf, C['panel'], rect, 0)
        cy = y + 14

        def label(text, cx, cy, font, col):
            surf.blit(font.render(text, True, col), (cx, cy))

        title = self.renderer.font_big.render('SOKOBAN', True, C['accent'])
        surf.blit(title, (x+w//2-title.get_width()//2, cy)); cy += 34

        lvl = self.levels[self.level_idx]
        label(f"Level {lvl['id']}: {lvl['name']}", x+15, cy,
              self.renderer.font_body, C['text_hi']); cy += 18
        label(f"Difficulty: {lvl.get('difficulty','?')}", x+15, cy,
              self.renderer.font_small, C['text_dim']); cy += 20

        div = pygame.Surface((w-30,1)); div.fill(C['wall_hi'])
        surf.blit(div, (x+15, cy)); cy += 10

        # Agent status boxes
        for arena, col, agent in [
            (self.left,  C['accent'],  self.pure_agent),
            (self.right, C['accent2'], self.hybrid_agent),
        ]:
            running = agent and agent.running
            sc = C['success'] if arena.state.is_solved() else \
                 C['accent2'] if running else C['text_dim']
            label(arena.label, x+15, cy, self.renderer.font_body, col); cy += 16
            label(arena.status[:38], x+15, cy, self.renderer.font_small, sc); cy += 18

        surf.blit(div, (x+15, cy)); cy += 10

        # Controls
        label('CONTROLS', x+15, cy, self.renderer.font_body, C['text_hi']); cy += 14
        speed = TOT_SPEEDS[self.tot_speed_idx]
        for key, desc in [
            ('G', 'Pure LLM agent'),
            ('T', 'Hybrid LLM+ToT'),
            ('B', 'Results table'),
            ('C/V', f'Speed: {speed}x'),
            ('A/D', 'Left board replay'),
            ('<-/->', 'Right board replay'),
            ('W/S', 'Left log scroll'),
            ('Up/Dn', 'Right log scroll'),
            ('R', 'Restart level'),
            ('N/P', 'Next/Prev level'),
            ('L', 'Select level'),
        ]:
            ks  = self.renderer.font_small.render(key, True, C['bg'])
            bw2 = ks.get_width()+8; bh2 = ks.get_height()+4
            draw_rounded_rect(surf, C['accent'], pygame.Rect(x+15,cy,bw2,bh2), 3)
            surf.blit(ks, (x+19, cy+2))
            label(desc, x+15+bw2+5, cy+2, self.renderer.font_small, C['text'])
            cy += bh2+3

    def _draw_results(self):
        from renderer import draw_rounded_rect
        sw, sh = self.screen.get_size()
        self.screen.fill(C['bg'])
        surf = self.screen

        FB = self.renderer.font_big
        FM = self.renderer.font_body
        FS = self.renderer.font_small

        # Title
        surf.blit(FB.render('BENCHMARK RESULTS', True, C['accent']), (20, 10))
        surf.blit(FS.render('B / Esc = close   R = rerun   1 = toggle A*   2 = toggle Pure LLM', True, C['text_dim']), (20, 44))

        # Toggle buttons
        astar_col = C['accent']  if self.bench_run_astar else C['error']
        pure_col  = C['success'] if self.bench_run_pure  else C['error']
        astar_lbl = '[A* ON]'  if self.bench_run_astar else '[A* OFF]'
        pure_lbl  = '[Pure LLM ON]' if self.bench_run_pure else '[Pure LLM OFF]'
        surf.blit(FM.render(astar_lbl, True, astar_col), (sw-280, 10))
        surf.blit(FM.render(pure_lbl,  True, pure_col),  (sw-280, 34))

        running_dots = '.' * (int(time.time()*3) % 4)

        # ── Table layout ──────────────────────────────────────────────────────
        LVL_X  = 18
        A_BASE = 300
        A_OK   = A_BASE;  A_STP = A_BASE+22; A_NOD = A_BASE+72; A_MS = A_BASE+132
        P_BASE = A_BASE + 185
        P_OK   = P_BASE;  P_STP = P_BASE+22; P_MS  = P_BASE+72
        H_BASE = P_BASE + 140
        H_OK   = H_BASE;  H_STP = H_BASE+22; H_BAK = H_BASE+72; H_MS = H_BASE+128
        H_ILL  = H_BASE+210; H_DL = H_BASE+265

        ROW_H = 26
        cy    = HDR_Y = 68

        def cell(txt, x, y, col, font=FS):
            surf.blit(font.render(str(txt), True, col), (x, y))

        # Agent headers
        surf.blit(FM.render('A*',             True, C['accent']),  (A_BASE, cy))
        surf.blit(FM.render('Pure LLM',       True, C['success']), (P_BASE, cy))
        surf.blit(FM.render('Hybrid LLM+ToT  [ill=Phi illegal  dl=Phi deadlock]', True, C['accent2']), (H_BASE, cy))
        cy += 20
        for x, lbl in [(A_STP,'steps'),(A_NOD,'nodes'),(A_MS,'ms'),
                       (P_STP,'steps'),(P_MS,'ms'),
                       (H_STP,'steps'),(H_BAK,'backs'),(H_MS,'ms'),(H_ILL,'ill'),(H_DL,'dl')]:
            cell(lbl, x, cy, C['text_dim'])
        cy += 18
        pygame.draw.line(surf, C['wall_hi'], (LVL_X, cy), (sw-20, cy), 1)
        cy += 6

        for level in self.levels:
            lid  = level['id']
            cell(f"{lid}. {level['name']}", LVL_X, cy+2, C['text'])

            for agent, ok_x, cols in [
                ('astar',    A_OK, [(A_STP,'steps'),(A_NOD,'nodes'),(A_MS,'ms_str')]),
                ('pure_llm', P_OK, [(P_STP,'steps'),(P_MS,'ms_str')]),
                ('hybrid',   H_OK, [(H_STP,'steps'),(H_BAK,'backtracks'),(H_MS,'ms_str'),(H_ILL,'illegal_rejected'),(H_DL,'deadlock_rejected')]),
            ]:
                res = self.bench.results.get((agent, lid))

                if res is None:
                    cell('…', ok_x, cy+2, C['text_dim'])
                elif res.get('status') == 'running...':
                    cell(f'run{running_dots}', ok_x, cy+2, C['accent2'])
                else:
                    solved = res.get('status') == 'solved'
                    ok_sym = '✓' if solved else '✗'
                    ok_col = C['success'] if solved else C['error']
                    cell(ok_sym, ok_x, cy+2, ok_col, FM)
                    if not solved:
                        cell('unsolved', ok_x+20, cy+2, C['error'])
                    else:
                        ms_str = f"{res.get('compute_ms','-')}ms"
                        r = dict(res, ms_str=ms_str)
                        for x, k in cols:
                            cell(r.get(k, '-'), x, cy+2, C['text_dim'])

            cy += ROW_H

        pygame.draw.line(surf, C['wall_hi'], (LVL_X, cy+4), (sw-20, cy+4), 1)
        cy += 14
        if self.bench.running:
            done  = sum(1 for v in self.bench.results.values()
                        if v.get('status') not in ('running...', None))
            total = len(self.levels) * 3
            foot  = f'Running… {done}/{total} complete'
            col   = C['accent2']
        else:
            n = len(self.levels)
            sa = sum(1 for (a,_),v in self.bench.results.items()
                     if a=='astar'    and v.get('status')=='solved')
            sp = sum(1 for (a,_),v in self.bench.results.items()
                     if a=='pure_llm' and v.get('status')=='solved')
            sh = sum(1 for (a,_),v in self.bench.results.items()
                     if a=='hybrid'   and v.get('status')=='solved')
            foot = f'Complete —  A*: {sa}/{n}   Pure LLM: {sp}/{n}   Hybrid: {sh}/{n}'
            col  = C['success']
        surf.blit(FM.render(foot, True, col), (LVL_X, cy))

        pygame.display.flip()

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            if self.mode not in ('results',):
                self.update(dt)
            self.draw()


if __name__ == '__main__':
    App().run()

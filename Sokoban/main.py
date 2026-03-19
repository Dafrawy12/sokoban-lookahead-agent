"""
sokoban/main.py
Dual-arena layout:
  LEFT  — A* solver  (Space to solve, Enter to play)
  RIGHT — ToT agent  (T to start/stop — fully automatic)
"""

from __future__ import annotations
import sys, time, pathlib, threading
from dataclasses import dataclass
from typing import Optional
import pygame

from game import GameState, load_levels, level_to_state, DIRECTIONS
from solver import solve, SolverResult, MoveStep, deadlock_type
from renderer import Renderer, TILE, PANEL_W, C
from tot_agent import ToTAgent

LEVELS_PATH = pathlib.Path(__file__).parent / 'levels' / 'levels.json'

FPS       = 60
SPEEDS     = [0.25, 0.5, 1.0, 2.0, 4.0]
TOT_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # ToT can go faster
MOVE_DUR  = 0.12
PUSH_DUR  = 0.18
PLAY_DUR  = 0.45
ARENA_GAP = 24
ARENA_PAD = 16


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


# ── Animation ─────────────────────────────────────────────────────────────────
@dataclass
class Anim:
    player_from: tuple[int, int]
    player_to:   tuple[int, int]
    box_from:    Optional[tuple[int, int]]
    box_to:      Optional[tuple[int, int]]
    t:           float
    duration:    float
    prev_state:  GameState
    next_state:  GameState

    @property
    def done(self) -> bool:
        return self.t >= 1.0

    def advance(self, dt: float):
        self.t = min(1.0, self.t + dt / self.duration)

    def alpha(self) -> float:
        return ease_out_cubic(self.t)


# ── Arena ─────────────────────────────────────────────────────────────────────
class Arena:
    def __init__(self, label: str, initial: GameState, level: dict):
        self.label   = label
        self.level   = level
        self.state   = initial
        self.anim: Optional[Anim] = None
        self.history: list[GameState] = []
        self.solver_result: Optional[SolverResult] = None
        self.solving       = False
        self.playing       = False
        self.playback_step = -1
        self.show_overlay  = True

    def reset(self, level: dict):
        self.level   = level
        self.state   = level_to_state(level)
        self.anim    = None
        self.history = []
        self.solver_result = None
        self.solving       = False
        self.playing       = False
        self.playback_step = -1

    def start_anim(self, prev, nxt, box_from=None, box_to=None, dur=MOVE_DUR):
        self.anim = Anim(
            player_from=prev.player, player_to=nxt.player,
            box_from=box_from, box_to=box_to,
            t=0.0, duration=dur,
            prev_state=prev, next_state=nxt,
        )

    def snap_anim(self):
        if self.anim and not self.anim.done:
            self.state = self.anim.next_state
        self.anim = None

    def advance_anim(self, dt: float):
        if self.anim:
            self.anim.advance(dt)
            if self.anim.done:
                self.anim = None

    def do_move(self, direction: str, speed: float = 1.0):
        self.snap_anim()
        prev = self.state
        new_state = prev.apply_move(direction)
        if not new_state:
            return False
        self.history.append(prev)
        dr, dc   = DIRECTIONS[direction]
        nr, nc   = prev.player[0] + dr, prev.player[1] + dc
        pushed   = (nr, nc) in prev.boxes
        box_from = (nr, nc)       if pushed else None
        box_to   = (nr+dr, nc+dc) if pushed else None
        dur      = (PUSH_DUR if pushed else MOVE_DUR) / speed
        self.state = new_state
        self.start_anim(prev, new_state, box_from, box_to, dur)
        return True

    def undo_move(self, dur: float = MOVE_DUR * 0.8):
        """Undo last move — used for backtracking animation."""
        if self.history:
            prev = self.history.pop()
            self.anim = Anim(
                player_from=self.state.player,
                player_to=prev.player,
                box_from=None, box_to=None,
                t=0.0, duration=dur,
                prev_state=self.state,
                next_state=prev,
            )
            self.state = prev

    def undo(self):
        if self.history:
            self.state = self.history.pop()
            self.anim  = None

    def step_forward(self, speed: float):
        sr = self.solver_result
        if not sr or not sr.solved:
            return
        total = len(sr.full_path)
        if self.playback_step >= total - 1:
            return
        prev_state = (level_to_state(self.level)
                      if self.playback_step < 0
                      else sr.full_path[self.playback_step].state_after)
        self.playback_step += 1
        step: MoveStep = sr.full_path[self.playback_step]
        base_dur = PLAY_DUR if step.is_push else PLAY_DUR * 0.4
        dur = max(0.04, base_dur / speed)
        self.state = step.state_after
        self.start_anim(prev_state, step.state_after,
                        box_from=step.box_from, box_to=step.box_to, dur=dur)

    def restart_playback(self):
        self.playback_step = -1
        self.state = level_to_state(self.level)
        self.anim  = None

    def update_playback(self, speed: float):
        if not self.playing or self.anim is not None:
            return
        sr = self.solver_result
        if not sr or not sr.solved:
            return
        total = len(sr.full_path)
        if self.playback_step < total - 1:
            self.step_forward(speed)
        else:
            self.playing = False


# ── App ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Sokoban  ·  A* vs ToT')

        self.levels     = load_levels(LEVELS_PATH)
        self.level_idx  = 0
        self.speed_idx  = 2
        self.mode       = 'game'
        self.select_idx = 0

        lvl = self.levels[0]
        self.left  = Arena('A*',  level_to_state(lvl), lvl)
        self.right = Arena('ToT', level_to_state(lvl), lvl)

        self._init_screen()

        icon = pygame.Surface((32, 32))
        icon.fill(C['box'])
        pygame.draw.rect(icon, C['box_solved'], (8, 8, 16, 16))
        pygame.display.set_icon(icon)

        self.clock    = pygame.time.Clock()
        self.renderer = Renderer(self.screen)

        # ToT agent
        self.agent = ToTAgent(
            on_step=self._on_agent_step,
            on_status=self._on_agent_status,
        )

        # ToT display state (updated by agent callbacks)
        self.tot_status       = 'Press T to start'
        self.tot_candidates:  list[dict] = []
        self.tot_backtracking = False
        self.tot_depth        = 0
        self.tot_nodes        = 0
        self.tot_backtracks   = 0

        # ToT speed (separate from A* playback speed)
        self.tot_speed_idx    = 2   # default 1.0x

        # Results for comparison (set when each side finishes)
        self.result_astar:    dict = {}
        self.result_tot:      dict = {}
        self._tot_algo_stats: dict = {}  # partial — algo stats before animation drains
        self._tot_start_time: float = 0.0

        # pending move from agent thread → applied in update()
        self._pending: Optional[tuple[str, list, bool]] = None
        self._pending_lock = threading.Lock()

    # ── agent callbacks (called from agent thread) ────────────────────────
    def _on_agent_step(self, direction: str, candidates: list, backtracking: bool):
        with self._pending_lock:
            self._pending = (direction, candidates, backtracking)

    def _on_agent_status(self, msg: str):
        self.tot_status = msg
        # Record algo stats immediately when agent finishes — these are accurate right now
        if msg.startswith('Solved') and not self.result_tot:
            self._tot_algo_stats = {
                'nodes':      self.agent.nodes_explored,
                'backtracks': self.agent.backtracks,
                'depth':      self.agent.current_depth,
                'time_ms':    self.agent.compute_ms,
            }
            # steps/pushes deferred — counted in update() once animation drains

    # ── layout ────────────────────────────────────────────────────────────
    def _board_size(self):
        state = level_to_state(self.levels[self.level_idx])
        return state.cols * TILE, state.rows * TILE

    def _init_screen(self):
        bw, bh = self._board_size()
        win_w = ARENA_PAD + bw + ARENA_GAP + bw + ARENA_PAD + PANEL_W
        win_h = max(600, bh + 80)
        if not hasattr(self, 'screen'):
            self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        self._calc_offsets()

    def _calc_offsets(self):
        sw, sh   = self.screen.get_size()
        bw, bh   = self._board_size()
        boards_w = bw * 2 + ARENA_GAP
        start_x  = max(ARENA_PAD, (sw - PANEL_W - boards_w) // 2)
        oy = max(40, (sh - bh) // 2)
        self.left_offset  = (start_x, oy)
        self.right_offset = (start_x + bw + ARENA_GAP, oy)

    # ── level init ────────────────────────────────────────────────────────
    def _init_level(self):
        self.agent.stop()
        lvl = self.levels[self.level_idx]
        self.left.reset(lvl)
        self.right.reset(lvl)
        self.tot_status       = 'Press T to start'
        self.tot_candidates   = []
        self.tot_backtracking = False
        self.tot_depth        = 0
        self.tot_nodes        = 0
        self.tot_backtracks   = 0
        self.result_astar     = {}
        self.result_tot       = {}
        self._tot_algo_stats  = {}
        with self._pending_lock:
            self._pending = None
        self._init_screen()

    # ── A* solver ─────────────────────────────────────────────────────────
    def _start_solver(self):
        if self.left.solving:
            return
        self.left.solving       = True
        self.left.solver_result = None
        lvl = self.left.level
        def _run():
            result = solve(level_to_state(lvl))
            self.left.solver_result = result
            self.left.solving       = False
            self.left.restart_playback()
            if result.solved:
                self.left.playing = True
                self.result_astar = {
                    'pushes':  result.num_pushes,
                    'steps':   len(result.full_path),
                    'nodes':   result.nodes_expanded,
                    'time_ms': result.runtime_ms,
                }
        threading.Thread(target=_run, daemon=True).start()

    # ── ToT toggle ────────────────────────────────────────────────────────
    def _toggle_agent(self):
        if self.agent.running:
            self.agent.stop()
            self.tot_status = 'Stopped'
        else:
            self.right.reset(self.right.level)
            self.tot_status       = 'Starting...'
            self.tot_candidates   = []
            self.tot_backtracking = False
            self.result_tot       = {}
            self._tot_start_time  = time.time()
            self.agent.start(lambda: self.right.state)

    # ── input ─────────────────────────────────────────────────────────────
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.VIDEORESIZE:
                self._calc_offsets()
            if self.mode == 'select':
                self._handle_select(ev)
            else:
                self._handle_game(ev)

    def _handle_game(self, ev):
        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key

        if k == pygame.K_l:
            self.mode = 'select'; self.select_idx = self.level_idx; return
        if k == pygame.K_r:
            self._init_level(); return
        if k == pygame.K_n:
            self.level_idx = (self.level_idx + 1) % len(self.levels)
            self._init_level(); return
        if k == pygame.K_p:
            self.level_idx = (self.level_idx - 1) % len(self.levels)
            self._init_level(); return
        if k == pygame.K_SPACE:
            self._start_solver(); return
        if k == pygame.K_o:
            self.left.show_overlay = not self.left.show_overlay; return
        if k == pygame.K_t:
            self._toggle_agent(); return

        # Speed controls — Z/X for A* playback, C/V for ToT
        if k == pygame.K_x:
            self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1); return
        if k == pygame.K_z:
            self.speed_idx = max(self.speed_idx - 1, 0); return
        if k == pygame.K_v:
            self.tot_speed_idx = min(self.tot_speed_idx + 1, len(TOT_SPEEDS) - 1); return
        if k == pygame.K_c:
            self.tot_speed_idx = max(self.tot_speed_idx - 1, 0); return

        # Left arena playback
        sr    = self.left.solver_result
        speed = SPEEDS[self.speed_idx]
        if sr and sr.solved:
            total = len(sr.full_path)
            if k == pygame.K_RETURN:
                if not self.left.playing and self.left.playback_step >= total - 1:
                    self.left.restart_playback()
                self.left.playing = not self.left.playing
            elif k == pygame.K_HOME:
                self.left.restart_playback()
            elif k == pygame.K_END:
                self.left.playback_step = total - 1
                self.left.state = sr.full_path[-1].state_after
                self.left.anim  = None
            elif k == pygame.K_q:
                self.left.restart_playback(); self.left.playing = True

    def _handle_select(self, ev):
        if ev.type != pygame.KEYDOWN:
            return
        k, n = ev.key, len(self.levels)
        if   k == pygame.K_RIGHT: self.select_idx = (self.select_idx + 1) % n
        elif k == pygame.K_LEFT:  self.select_idx = (self.select_idx - 1) % n
        elif k == pygame.K_DOWN:  self.select_idx = (self.select_idx + 4) % n
        elif k == pygame.K_UP:    self.select_idx = (self.select_idx - 4) % n
        elif k in (pygame.K_RETURN, pygame.K_SPACE):
            self.level_idx = self.select_idx
            self._init_level(); self.mode = 'game'
        elif k == pygame.K_ESCAPE:
            self.mode = 'game'

    # ── update ────────────────────────────────────────────────────────────
    def update(self, dt: float):
        speed = SPEEDS[self.speed_idx]
        self.left.advance_anim(dt)
        self.right.advance_anim(dt)
        self.left.update_playback(speed)

        # Apply pending ToT move when animation is free
        if self.right.anim is None:
            with self._pending_lock:
                pending = self._pending
                self._pending = None

            if pending is not None:
                direction, candidates, backtracking = pending
                self.tot_candidates   = candidates
                self.tot_backtracking = backtracking
                self.tot_depth        = self.agent.current_depth
                self.tot_nodes        = self.agent.nodes_explored
                self.tot_backtracks   = self.agent.backtracks

                tot_speed = TOT_SPEEDS[self.tot_speed_idx]

                if backtracking:
                    self.right.undo_move(dur=MOVE_DUR * 0.8 / tot_speed)
                else:
                    self.right.do_move(direction, speed=tot_speed)

                self.agent.signal_move_done()

        # Finalize ToT results once animation fully drained and board is solved
        if (self._tot_algo_stats
                and not self.result_tot
                and not self.agent.running
                and self.right.anim is None
                and self._pending is None
                and self.right.state.is_solved()):

            # Count steps and pushes from the completed history
            history = self.right.history
            pushes  = sum(
                1 for prev, curr in zip(
                    [level_to_state(self.right.level)] + history[:-1],
                    history
                )
                if prev.boxes != curr.boxes
            )
            self.result_tot = {
                **self._tot_algo_stats,
                'pushes': pushes,
                'steps':  len(history),
            }
            self._tot_algo_stats = {}   # consumed

    # ── draw ──────────────────────────────────────────────────────────────
    def draw(self):
        sw, sh = self.screen.get_size()
        self.screen.fill(C['bg'])
        self._calc_offsets()

        bw, bh   = self._board_size()
        lox, loy = self.left_offset
        rox, roy = self.right_offset

        # labels
        lbl_l = self.renderer.font_title.render('A*  Solver', True, C['accent'])
        lbl_r = self.renderer.font_title.render('ToT Agent',  True, C['accent2'])
        self.screen.blit(lbl_l, (lox, 10))
        self.screen.blit(lbl_r, (rox, 10))

        # left board
        left_overlay = None
        if self.left.show_overlay and self.left.solver_result and self.left.solver_result.solved:
            left_overlay = self.left.solver_result.push_sequence
        self.renderer.draw_board(
            state=self.left.state, offset=self.left_offset,
            overlay_steps=left_overlay, current_step=self.left.playback_step,
            show_overlay=self.left.show_overlay, anim=self.left.anim,
        )

        # right board
        self.renderer.draw_board(
            state=self.right.state, offset=self.right_offset,
            overlay_steps=None, current_step=-1,
            show_overlay=False, anim=self.right.anim,
        )

        # divider
        div_x = rox - ARENA_GAP // 2
        pygame.draw.line(self.screen, C['wall_hi'], (div_x, 0), (div_x, sh), 1)

        # panel
        panel_rect = pygame.Rect(sw - PANEL_W, 0, PANEL_W, sh)
        pygame.draw.line(self.screen, C['wall_hi'], (sw - PANEL_W, 0), (sw - PANEL_W, sh), 1)
        self.renderer.draw_dual_panel(
            panel_rect=panel_rect,
            level_info=self.left.level,
            left=self.left,
            right=self.right,
            speed=SPEEDS[self.speed_idx],
            tot_speed=TOT_SPEEDS[self.tot_speed_idx],
            tot_status=self.tot_status,
            tot_candidates=self.tot_candidates,
            tot_backtracking=self.tot_backtracking,
            tot_depth=self.tot_depth,
            tot_nodes=self.tot_nodes,
            tot_backtracks=self.tot_backtracks,
            agent_running=self.agent.running,
            result_astar=self.result_astar,
            result_tot=self.result_tot,
        )

        # A* spinner
        if self.left.solving:
            dots = '.' * (int(time.time() * 3) % 4)
            txt  = self.renderer.font_title.render(f'Solving{dots}', True, C['accent2'])
            self.screen.blit(txt, (lox, sh - 34))

        # backtracking flash on right arena
        if self.tot_backtracking and self.agent.running:
            flash = pygame.Surface((bw, bh), pygame.SRCALPHA)
            flash.fill((*C['error'], 25))
            self.screen.blit(flash, (rox, roy))

        # deadlock banners
        self._draw_deadlock_banner(self.left,  bw, sh)
        self._draw_deadlock_banner(self.right, bw, sh)

        # win overlays
        self._draw_win_overlay(self.left.state,  self.left_offset,  bw, bh, C['accent'])
        self._draw_win_overlay(self.right.state, self.right_offset, bw, bh, C['accent2'])

        if self.mode == 'select':
            self.renderer.draw_level_select(self.levels, self.select_idx, sw, sh)

        pygame.display.flip()

    def _draw_deadlock_banner(self, arena: Arena, bw: int, sh: int):
        if arena.state.is_solved() or arena.solving:
            return
        dl = deadlock_type(arena.state)
        if not dl:
            return
        ox, _ = self.left_offset if arena is self.left else self.right_offset
        banner_h = 36
        by = sh - banner_h - 8
        s = pygame.Surface((bw, banner_h), pygame.SRCALPHA)
        s.fill((*C['error'], 190))
        pygame.draw.rect(s, C['error'], (0, 0, bw, banner_h), 2, border_radius=5)
        self.screen.blit(s, (ox, by))
        txt = self.renderer.font_body.render(f'  {dl}', True, (255, 220, 220))
        self.screen.blit(txt, txt.get_rect(center=(ox + bw//2, by + banner_h//2)))

    def _draw_win_overlay(self, state, offset, bw, bh, color):
        if not state.is_solved():
            return
        ox, oy = offset
        s = pygame.Surface((bw, bh), pygame.SRCALPHA)
        s.fill((*color, 30))
        self.screen.blit(s, (ox, oy))
        txt = self.renderer.font_title.render('  SOLVED', True, color)
        self.screen.blit(txt, txt.get_rect(center=(ox + bw//2, oy + bh//2)))

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()


if __name__ == '__main__':
    App().run()

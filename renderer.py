"""
sokoban/renderer.py
All Pygame drawing code. Stateless drawing functions + a Renderer class
that owns fonts and cached surfaces.

Color palette  — dark industrial theme:
  BG            #0d0f14   near-black
  PANEL         #161920   slightly lighter
  WALL          #2e3440   slate
  FLOOR         #1a1d24   subtle
  TARGET        #4c566a   muted blue-grey ring
  BOX           #bf6a2e   amber
  BOX_SOLVED    #5e9e3e   green
  PLAYER        #81a1c1   icy blue
  PATH_OVERLAY  #ebcb8b   gold
  ACCENT        #88c0d0   teal highlight
  TEXT_MAIN     #eceff4   near-white
  TEXT_DIM      #4c566a   grey
"""

from __future__ import annotations
import pygame
import math
from typing import Optional
from game import GameState, TARGET, WALL

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    'bg':          (13,  15,  20),
    'panel':       (22,  25,  32),
    'panel_hi':    (30,  34,  44),
    'wall':        (46,  52,  64),
    'wall_hi':     (62,  70,  88),
    'floor':       (20,  22,  30),
    'floor_alt':   (24,  27,  36),
    'target':      (76,  86, 106),
    'target_ring': (136,192,208),
    'box':         (191,106, 46),
    'box_solved':  ( 94,158, 62),
    'box_shadow':  (100, 55, 20),
    'player':      (129,161,193),
    'player_hi':   (163,190,215),
    'path':        (235,203,139),
    'path_dim':    (120,100, 60),
    'accent':      (136,192,208),
    'accent2':     (180,142,173),
    'text':        (236,239,244),
    'text_dim':    ( 76, 86,106),
    'text_hi':     (136,192,208),
    'success':     ( 94,158, 62),
    'warning':     (191,106, 46),
    'error':       (191, 97,106),
    'button':      ( 38, 44, 58),
    'button_hi':   ( 55, 62, 80),
    'button_act':  ( 76, 86,106),
}

TILE = 56          # base tile size in pixels
PANEL_W = 320      # right panel width


# ── Helpers ───────────────────────────────────────────────────────────────────
def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i]-a[i])*t) for i in range(3))

def draw_rounded_rect(surf, color, rect, radius=6, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)

def draw_shadow(surf, rect, offset=4, alpha=80):
    s = pygame.Surface((rect.width + offset, rect.height + offset), pygame.SRCALPHA)
    pygame.draw.rect(s, (0,0,0,alpha), (offset, offset, rect.width, rect.height), border_radius=8)
    surf.blit(s, (rect.x, rect.y))


# ── Renderer class ────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        pygame.font.init()
        try:
            self.font_title  = pygame.font.SysFont('consolas,monospace', 22, bold=True)
            self.font_body   = pygame.font.SysFont('consolas,monospace', 17)
            self.font_small  = pygame.font.SysFont('consolas,monospace', 15)
            self.font_big    = pygame.font.SysFont('consolas,monospace', 30, bold=True)
            self.font_num    = pygame.font.SysFont('consolas,monospace', 13, bold=True)
        except:
            self.font_title  = pygame.font.Font(None, 26)
            self.font_body   = pygame.font.Font(None, 20)
            self.font_small  = pygame.font.Font(None, 18)
            self.font_big    = pygame.font.Font(None, 34)
            self.font_num    = pygame.font.Font(None, 16)

        self._tile_cache: dict[str, pygame.Surface] = {}

    # ── tile drawing ──────────────────────────────────────────────────────
    def _tile(self, key: str) -> pygame.Surface:
        if key in self._tile_cache:
            return self._tile_cache[key]
        s = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        self._draw_tile(s, key)
        self._tile_cache[key] = s
        return s

    def _draw_tile(self, s: pygame.Surface, key: str):
        T = TILE
        if key == 'wall':
            s.fill(C['wall'])
            # subtle bevel
            pygame.draw.line(s, C['wall_hi'], (0,0), (T,0), 2)
            pygame.draw.line(s, C['wall_hi'], (0,0), (0,T), 2)
            pygame.draw.line(s, (20,22,28), (T-1,0), (T-1,T), 1)
            pygame.draw.line(s, (20,22,28), (0,T-1), (T,T-1), 1)
            # grid lines
            pygame.draw.rect(s, C['wall_hi'], (2,2,T-4,T-4), 1)
        elif key == 'floor':
            s.fill(C['floor'])
            # subtle dot pattern
            for dr in range(0, T, 14):
                for dc in range(0, T, 14):
                    pygame.draw.circle(s, C['floor_alt'], (dc+7, dr+7), 1)
        elif key == 'target':
            s.fill(C['floor'])
            # concentric rings
            cx, cy = T//2, T//2
            pygame.draw.circle(s, C['target'], (cx,cy), T//2-6, 3)
            pygame.draw.circle(s, C['target_ring'], (cx,cy), T//2-12, 2)
            pygame.draw.circle(s, C['target_ring'], (cx,cy), 4)
        elif key == 'box':
            s.fill((0,0,0,0))
            pad = 5
            r = pygame.Rect(pad, pad, T-2*pad, T-2*pad)
            # shadow
            draw_shadow(s, r, 3, 120)
            # main face
            draw_rounded_rect(s, C['box'], r, 6)
            # highlight top
            draw_rounded_rect(s, lerp_color(C['box'],(255,220,180),0.25),
                              pygame.Rect(pad+3, pad+3, T-2*pad-6, 8), 4)
            # border
            draw_rounded_rect(s, C['box_shadow'], r, 6, 2, C['box_shadow'])
            # $ symbol
            txt = self.font_title.render('$', True, (255,220,160))
            s.blit(txt, txt.get_rect(center=(T//2, T//2)))
        elif key == 'box_solved':
            s.fill((0,0,0,0))
            pad = 5
            r = pygame.Rect(pad, pad, T-2*pad, T-2*pad)
            draw_shadow(s, r, 3, 120)
            draw_rounded_rect(s, C['box_solved'], r, 6)
            draw_rounded_rect(s, lerp_color(C['box_solved'],(200,255,180),0.3),
                              pygame.Rect(pad+3, pad+3, T-2*pad-6, 8), 4)
            txt = self.font_title.render('✓', True, (200,255,160))
            s.blit(txt, txt.get_rect(center=(T//2, T//2)))
        elif key == 'player':
            s.fill((0,0,0,0))
            cx, cy = T//2, T//2
            # glow
            glow = pygame.Surface((T, T), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*C['player'], 40), (cx,cy), T//2-2)
            s.blit(glow, (0,0))
            # body circle
            pygame.draw.circle(s, C['player'], (cx, cy+2), T//2-8)
            pygame.draw.circle(s, C['player_hi'], (cx-3, cy-2), T//2-14)
            # direction indicator dot
            pygame.draw.circle(s, (255,255,255), (cx, cy-4), 3)

    def draw_board(self, state: GameState, offset: tuple[int,int],
                   overlay_steps: list | None = None,
                   current_step: int = -1,
                   show_overlay: bool = True,
                   anim=None):
        ox, oy = offset
        T = TILE

        # During animation draw tiles from prev_state so background is correct
        draw_state = anim.prev_state if anim and not anim.done else state

        # ── base tiles ──
        for r in range(draw_state.rows):
            for c in range(draw_state.cols):
                cell = draw_state.grid[r][c]
                x, y = ox + c*T, oy + r*T
                if cell == '#':
                    self.screen.blit(self._tile('wall'), (x, y))
                elif cell == '.':
                    self.screen.blit(self._tile('target'), (x, y))
                else:
                    self.screen.blit(self._tile('floor'), (x, y))

        # ── path overlay (below pieces) ──
        if show_overlay and overlay_steps:
            self._draw_path_overlay(state, offset, overlay_steps, current_step)

        # ── boxes ──
        if anim and not anim.done:
            a = anim.alpha()
            # Draw all stationary boxes
            for (br, bc) in anim.prev_state.boxes:
                if anim.box_from and (br, bc) == anim.box_from:
                    continue   # skip the animated one
                key = 'box_solved' if anim.prev_state.grid[br][bc] == '.' else 'box'
                self.screen.blit(self._tile(key), (ox+bc*T, oy+br*T))
            # Animated box — lerped pixel position
            if anim.box_from and anim.box_to:
                fr, fc = anim.box_from
                tr, tc = anim.box_to
                bx = ox + (fc + (tc - fc) * a) * T
                by = oy + (fr + (tr - fr) * a) * T
                key = 'box_solved' if anim.next_state.grid[tr][tc] == '.' else 'box'
                self.screen.blit(self._tile(key), (bx, by))
        else:
            for (br, bc) in state.boxes:
                key = 'box_solved' if state.grid[br][bc] == '.' else 'box'
                self.screen.blit(self._tile(key), (ox+bc*T, oy+br*T))

        # ── player ──
        if anim and not anim.done:
            a = anim.alpha()
            fr, fc = anim.player_from
            tr, tc = anim.player_to
            px = ox + (fc + (tc - fc) * a) * T
            py = oy + (fr + (tr - fr) * a) * T
            # squash & stretch: slight squish mid-travel
            dr = tr - fr
            dc = tc - fc
            squash = abs(1.0 - abs(a - 0.5) * 2)   # peaks at t=0.5
            sx = 1.0 - abs(dc) * 0.12 * squash
            sy = 1.0 - abs(dr) * 0.12 * squash
            tile = self._tile('player')
            if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
                w = max(1, int(T * sx))
                h = max(1, int(T * sy))
                tile = pygame.transform.scale(tile, (w, h))
                px += (T - w) / 2
                py += (T - h) / 2
            self.screen.blit(tile, (px, py))
        else:
            pr, pc = state.player
            self.screen.blit(self._tile('player'), (ox+pc*T, oy+pr*T))

    def _draw_path_overlay(self, state, offset, steps, current_step):
        ox, oy = offset
        T = TILE

        for i, step in enumerate(steps):
            is_done    = i < current_step
            is_current = i == current_step

            br, bc = step.box_from
            tr, tc = step.box_to

            if is_done:
                color = (*C['path_dim'], 100)
                num_color = C['path_dim']
            elif is_current:
                color = (*C['path'], 200)
                num_color = C['path']
            else:
                alpha = max(40, 140 - i * 8)
                color = (*C['path'], alpha)
                num_color = lerp_color(C['path_dim'], C['path'], 0.4)

            # destination cell highlight
            rect = pygame.Rect(ox+tc*T+3, oy+tr*T+3, T-6, T-6)
            s = pygame.Surface((T-6, T-6), pygame.SRCALPHA)
            s.fill((*num_color, 60) if not is_current else (*C['path'], 80))
            self.screen.blit(s, (ox+tc*T+3, oy+tr*T+3))
            pygame.draw.rect(self.screen, (*num_color, 180), rect, 2, border_radius=4)

            # arrow from box_from → box_to
            fx = ox + bc*T + T//2
            fy = oy + br*T + T//2
            tx2 = ox + tc*T + T//2
            ty2 = oy + tr*T + T//2
            a = max(60, 180 - i*10) if not is_done else 40
            self._draw_arrow(fx, fy, tx2, ty2, (*num_color, a), 2 if is_current else 1)

            # step number badge
            num_txt = self.font_num.render(str(i+1), True, C['bg'])
            badge_r = max(10, num_txt.get_width()//2 + 4)
            badge_surf = pygame.Surface((badge_r*2, badge_r*2), pygame.SRCALPHA)
            pygame.draw.circle(badge_surf, (*num_color, 220), (badge_r, badge_r), badge_r)
            badge_surf.blit(num_txt, num_txt.get_rect(center=(badge_r, badge_r)))
            self.screen.blit(badge_surf, (ox+tc*T + T//2 - badge_r, oy+tr*T + T//2 - badge_r))

    def _draw_arrow(self, x1, y1, x2, y2, color, width=1):
        if x1 == x2 and y1 == y2:
            return
        surf = self.screen
        pygame.draw.line(surf, color[:3], (x1,y1), (x2,y2), width)
        # arrowhead
        import math
        angle = math.atan2(y2-y1, x2-x1)
        size = 8
        ax1 = x2 - size*math.cos(angle-0.4)
        ay1 = y2 - size*math.sin(angle-0.4)
        ax2 = x2 - size*math.cos(angle+0.4)
        ay2 = y2 - size*math.sin(angle+0.4)
        pygame.draw.polygon(surf, color[:3], [(x2,y2),(ax1,ay1),(ax2,ay2)])

    # ── Panel (right side) ────────────────────────────────────────────────
    def draw_panel(self, panel_rect: pygame.Rect, level_info: dict,
                   solver_result, ui_state: dict):
        surf = self.screen
        x, y, w, h = panel_rect.x, panel_rect.y, panel_rect.width, panel_rect.height

        # background
        draw_rounded_rect(surf, C['panel'], panel_rect, 0)

        cy = y + 18

        # ── Title ──
        title = self.font_big.render('SOKOBAN', True, C['accent'])
        surf.blit(title, (x + w//2 - title.get_width()//2, cy))
        cy += 38

        line = pygame.Surface((w-30, 1))
        line.fill(C['wall_hi'])
        surf.blit(line, (x+15, cy))
        cy += 12

        # ── Level info ──
        self._label(surf, f"LEVEL {level_info.get('id','?')}", x+15, cy, self.font_title, C['text'])
        cy += 22
        diff = level_info.get('difficulty','').upper()
        diff_c = {'EASY': C['success'], 'MEDIUM': C['warning'], 'HARD': C['error']}.get(diff, C['text_dim'])
        self._label(surf, level_info.get('name',''), x+15, cy, self.font_body, C['text_dim'])
        self._label(surf, diff, x+w-15-self.font_body.size(diff)[0], cy, self.font_body, diff_c)
        cy += 24

        line2 = pygame.Surface((w-30, 1))
        line2.fill(C['panel_hi'])
        surf.blit(line2, (x+15, cy))
        cy += 12

        # ── Controls help ──
        self._label(surf, 'CONTROLS', x+15, cy, self.font_body, C['text_hi'])
        cy += 20
        controls = [
            ('Arrow / WASD', 'Move player'),
            ('R',            'Restart level'),
            ('U',            'Undo move'),
            ('Space',        'Run A* solver'),
            ('O',            'Toggle overlay'),
            ('Enter',        'Play / Pause'),
            ('Q',            'Restart playback'),
            ('← →',          'Step path'),
            ('Z / X',        'Speed down/up'),
            ('N / P',        'Next / Prev level'),
            ('L',            'Level select'),
        ]
        for key, desc in controls:
            # key badge
            key_surf = self.font_body.render(key, True, C['bg'])
            badge_w  = key_surf.get_width() + 10
            badge_h  = key_surf.get_height() + 4
            badge_r  = pygame.Rect(x+15, cy, badge_w, badge_h)
            draw_rounded_rect(surf, C['accent'], badge_r, 4)
            surf.blit(key_surf, (x+15+5, cy+2))
            self._label(surf, desc, x+15+badge_w+8, cy+2, self.font_body, C['text'])
            cy += badge_h + 5
        cy += 6

        line3 = pygame.Surface((w-30, 1))
        line3.fill(C['panel_hi'])
        surf.blit(line3, (x+15, cy))
        cy += 12

        # ── Solver stats ──
        self._label(surf, 'SOLVER', x+15, cy, self.font_body, C['text_hi'])
        cy += 20
        if solver_result is None:
            self._label(surf, 'Press SPACE to solve', x+15, cy, self.font_body, C['text'])
            cy += 18
        else:
            sr = solver_result
            if sr.solved:
                status_c = C['success']
                status_t = '✓ SOLVED'
            else:
                status_c = C['error']
                status_t = '✗ NO SOLUTION'
            self._label(surf, status_t, x+15, cy, self.font_title, status_c)
            cy += 22
            stats = [
                ('Pushes',    str(sr.num_pushes)),
                ('Expanded',  f"{sr.nodes_expanded:,}"),
                ('Generated', f"{sr.nodes_generated:,}"),
                ('Time',      f"{sr.runtime_ms:.1f} ms"),
            ]
            for label, val in stats:
                self._label(surf, label, x+15, cy, self.font_body, C['text_dim'])
                self._label(surf, val, x+w-15-self.font_body.size(val)[0], cy,
                           self.font_body, C['text'])
                cy += 18
        cy += 6

        line4 = pygame.Surface((w-30, 1))
        line4.fill(C['panel_hi'])
        surf.blit(line4, (x+15, cy))
        cy += 12

        # ── Playback controls ──
        if solver_result and solver_result.solved:
            self._label(surf, 'PLAYBACK', x+15, cy, self.font_body, C['text_hi'])
            cy += 20
            step = ui_state.get('step', -1)
            total = ui_state.get('total_steps', 0)
            playing = ui_state.get('playing', False)
            speed = ui_state.get('speed', 1.0)
            overlay = ui_state.get('show_overlay', True)

            step_str = f"Step: {step+1} / {total}" if total else "—"
            self._label(surf, step_str, x+15, cy, self.font_title, C['text'])
            cy += 24

            speed_str = f"Speed: {speed}x"
            ov_str    = f"Overlay: {'ON' if overlay else 'OFF'}"
            self._label(surf, speed_str, x+15, cy, self.font_body, C['path'])
            self._label(surf, ov_str, x+w-15-self.font_body.size(ov_str)[0], cy,
                        self.font_body, C['success'] if overlay else C['text_dim'])
            cy += 20

            play_str = '⏸ PAUSE' if playing else '▶ PLAY'
            self._label(surf, play_str + '  (ENTER)', x+15, cy, self.font_body, C['accent2'])
            cy += 18
            self._label(surf, '←/→ Step  |  Q Restart  |  Z/X Speed', x+15, cy, self.font_body, C['text'])
            cy += 18

        # ── Win message ──
        if ui_state.get('won', False):
            cy = y + h - 80
            banner = pygame.Surface((w, 60), pygame.SRCALPHA)
            banner.fill((*C['success'], 40))
            surf.blit(banner, (x, cy))
            pygame.draw.line(surf, C['success'], (x,cy), (x+w,cy), 2)
            win_txt = self.font_title.render('🎉  LEVEL COMPLETE!', True, C['success'])
            surf.blit(win_txt, win_txt.get_rect(center=(x+w//2, cy+30)))

    def _label(self, surf, text, x, y, font, color):
        t = font.render(text, True, color)
        surf.blit(t, (x, y))

    def draw_level_select(self, levels: list[dict], selected: int, screen_w: int, screen_h: int):
        """Draw a level selection overlay."""
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        overlay.fill((5, 7, 12, 220))
        self.screen.blit(overlay, (0,0))

        title = self.font_big.render('SELECT LEVEL', True, C['accent'])
        self.screen.blit(title, title.get_rect(center=(screen_w//2, 60)))

        cols = 4
        card_w, card_h = 160, 90
        margin = 20
        start_x = screen_w//2 - (cols*(card_w+margin))//2
        start_y = 110

        for i, lvl in enumerate(levels):
            row = i // cols
            col = i % cols
            cx = start_x + col*(card_w+margin)
            cy = start_y + row*(card_h+margin)
            r = pygame.Rect(cx, cy, card_w, card_h)
            is_sel = (i == selected)
            bg = C['button_act'] if is_sel else C['button']
            border = C['accent'] if is_sel else C['panel_hi']
            draw_shadow(self.screen, r, 4, 100)
            draw_rounded_rect(self.screen, bg, r, 8, 2, border)

            n = self.font_title.render(f"#{lvl['id']}", True, C['text_hi'] if is_sel else C['text'])
            self.screen.blit(n, n.get_rect(center=(cx+card_w//2, cy+22)))
            nm = self.font_small.render(lvl['name'], True, C['text'] if is_sel else C['text_dim'])
            self.screen.blit(nm, nm.get_rect(center=(cx+card_w//2, cy+44)))
            diff = lvl.get('difficulty','').upper()
            dc = {'EASY': C['success'], 'MEDIUM': C['warning'], 'HARD': C['error']}.get(diff, C['text_dim'])
            dt = self.font_small.render(diff, True, dc)
            self.screen.blit(dt, dt.get_rect(center=(cx+card_w//2, cy+62)))

        hint = self.font_body.render('Arrow keys to navigate · Enter to select · Esc to cancel', True, C['text_dim'])
        self.screen.blit(hint, hint.get_rect(center=(screen_w//2, screen_h-40)))

    # ── Dual panel ────────────────────────────────────────────────────────
    def draw_dual_panel(self, panel_rect, level_info, left, right, speed, tot_speed,
                        tot_status, tot_candidates, tot_backtracking,
                        tot_depth, tot_nodes, tot_backtracks, agent_running,
                        result_astar, result_tot,
                        llm_status='', llm_reasoning='', llm_candidates=None,
                        llm_running=False):
        surf = self.screen
        x, y, w, h = panel_rect.x, panel_rect.y, panel_rect.width, panel_rect.height
        draw_rounded_rect(surf, C["panel"], panel_rect, 0)
        cy = y + 14

        title = self.font_big.render("SOKOBAN", True, C["accent"])
        surf.blit(title, (x + w//2 - title.get_width()//2, cy)); cy += 36
        div = pygame.Surface((w-30, 1)); div.fill(C["wall_hi"])
        surf.blit(div, (x+15, cy)); cy += 10

        lvl_id   = level_info.get("id", "?")
        lvl_name = level_info.get("name", "")
        diff     = level_info.get("difficulty", "").upper()
        diff_c   = {"EASY": C["success"], "MEDIUM": C["warning"], "HARD": C["error"]}.get(diff, C["text_dim"])
        self._label(surf, f"LEVEL {lvl_id}  {lvl_name}", x+15, cy, self.font_title, C["text"])
        self._label(surf, diff, x+w-15-self.font_body.size(diff)[0], cy, self.font_body, diff_c)
        cy += 22
        div2 = pygame.Surface((w-30, 1)); div2.fill(C["panel_hi"])
        surf.blit(div2, (x+15, cy)); cy += 10

        # ── Live stats: two columns ──────────────────────────────────────
        col_w = (w-30)//2
        self._label(surf, "A*  Solver", x+15,       cy, self.font_body, C["accent"])
        self._label(surf, "ToT Agent",  x+15+col_w, cy, self.font_body, C["accent2"])
        cy += 18

        # A* live stats
        sr = left.solver_result
        cy_l = cy
        if left.solving:
            self._label(surf, "Solving...", x+15, cy_l, self.font_small, C["text_dim"]); cy_l += 14
        elif sr and sr.solved:
            for lbl, val in [("Pushes", str(sr.num_pushes)),("Nodes", f"{sr.nodes_expanded:,}"),("Time", f"{sr.runtime_ms:.0f}ms")]:
                self._label(surf, lbl, x+15,    cy_l, self.font_small, C["text_dim"])
                self._label(surf, val, x+15+52, cy_l, self.font_small, C["text"]); cy_l += 14
        else:
            self._label(surf, "Press Space", x+15, cy_l, self.font_small, C["text_dim"]); cy_l += 14

        # ToT live stats
        cy_r = cy
        status_col = (C["success"] if right.state.is_solved() else
                      C["error"]   if tot_backtracking else
                      C["accent2"] if agent_running else C["text_dim"])
        max_s = (w-30-col_w)//7
        self._label(surf, tot_status[:max_s], x+15+col_w, cy_r, self.font_small, status_col); cy_r += 14
        if agent_running or tot_nodes > 0:
            self._label(surf, f"D:{tot_depth} N:{tot_nodes:,}",    x+15+col_w, cy_r, self.font_small, C["text"]); cy_r += 14
            bt_c = C["error"] if tot_backtracks > 0 else C["text"]
            self._label(surf, f"Backtracks: {tot_backtracks}", x+15+col_w, cy_r, self.font_small, bt_c); cy_r += 14

        cy = max(cy_l, cy_r) + 6
        div3 = pygame.Surface((w-30, 1)); div3.fill(C["panel_hi"])
        surf.blit(div3, (x+15, cy)); cy += 8

        # ── Candidates ──────────────────────────────────────────────────
        lbl_c = C["error"] if tot_backtracking else C["text_hi"]
        lbl_t = "BACKTRACKING" if tot_backtracking else "CANDIDATES"
        self._label(surf, lbl_t, x+15, cy, self.font_body, lbl_c); cy += 16
        SC = {"valid": C["success"], "deadlock": C["error"], "illegal": C["text_dim"]}
        SI = {"valid": "OK", "deadlock": "DL", "illegal": "--"}
        if tot_candidates:
            for c in tot_candidates:
                move=c.get("move","?"); status=c.get("status","illegal")
                detail=c.get("detail",""); score=c.get("score")
                sc=SC.get(status,C["text_dim"]); icon=SI.get(status,"?")
                score_str = f"{score:.0f}" if score is not None else ""
                bt = self.font_small.render(f"{icon} {move}", True, C["bg"])
                bw2=bt.get_width()+10; bh2=bt.get_height()+6
                draw_rounded_rect(surf, sc, pygame.Rect(x+15, cy, bw2, bh2), 3)
                surf.blit(bt, (x+20, cy+3))
                info=(score_str+" "+detail).strip()
                mc=(w-40-bw2)//7
                self._label(surf, info[:mc], x+15+bw2+5, cy+3, self.font_small, sc)
                cy += bh2+3
        else:
            self._label(surf, "Press T to start", x+15, cy, self.font_small, C["text_dim"]); cy += 16

        cy += 4
        div4 = pygame.Surface((w-30, 1)); div4.fill(C["panel_hi"])
        surf.blit(div4, (x+15, cy)); cy += 8

        # ── Results comparison ───────────────────────────────────────────
        self._label(surf, "RESULTS", x+15, cy, self.font_body, C["text_hi"]); cy += 16

        # Header row
        self._label(surf, "Metric",  x+15,       cy, self.font_small, C["text_dim"])
        self._label(surf, "A*",      x+15+col_w, cy, self.font_small, C["accent"])
        self._label(surf, "ToT",     x+15+col_w+(col_w//2), cy, self.font_small, C["accent2"])
        cy += 14

        def cmp_row(label, a_val, t_val, lower_better=True):
            nonlocal cy
            self._label(surf, label, x+15, cy, self.font_small, C["text_dim"])
            # Color the winner green, loser red (only if both have values)
            if a_val is not None and t_val is not None:
                try:
                    a_n = float(str(a_val).replace(",","").replace("ms","").strip())
                    t_n = float(str(t_val).replace(",","").replace("ms","").strip())
                    if lower_better:
                        a_c = C["success"] if a_n <= t_n else C["error"]
                        t_c = C["success"] if t_n <= a_n else C["error"]
                    else:
                        a_c = C["success"] if a_n >= t_n else C["error"]
                        t_c = C["success"] if t_n >= a_n else C["error"]
                except:
                    a_c = t_c = C["text"]
            else:
                a_c = t_c = C["text_dim"]

            a_str = str(a_val) if a_val is not None else "—"
            t_str = str(t_val) if t_val is not None else "—"
            self._label(surf, a_str, x+15+col_w,            cy, self.font_small, a_c)
            self._label(surf, t_str, x+15+col_w+(col_w//2), cy, self.font_small, t_c)
            cy += 14

        ra = result_astar; rt = result_tot

        # Steps = total player moves (incl walks). Pushes = box pushes only (solution quality).
        cmp_row("Pushes ★",
                ra.get("pushes") if ra else None,
                rt.get("pushes") if rt else None)
        cmp_row("Tot steps",
                ra.get("steps") if ra else None,
                rt.get("steps") if rt else None)
        cmp_row("Nodes",
                f"{ra['nodes']:,}" if ra else None,
                f"{rt['nodes']:,}" if rt else None)
        cmp_row("Backtracks",
                "0" if ra else None,
                str(rt.get("backtracks", "—")) if rt else None)
        cmp_row("Phi illegal",
                "—" if ra else None,
                str(rt.get("illegal_rejected", "—")) if rt else None)
        cmp_row("Phi deadlock",
                "—" if ra else None,
                str(rt.get("deadlock_rejected", "—")) if rt else None)
        cmp_row("Compute ms",
                f"{ra['time_ms']:.1f}" if ra else None,
                f"{rt['time_ms']:.1f}" if rt else None)

        if not ra and not rt:
            self._label(surf, "Run both to compare", x+15, cy-14, self.font_small, C["text_dim"])
        elif ra and not rt:
            self._label(surf, "Press T to run ToT", x+15, cy, self.font_small, C["text_dim"]); cy += 14
        elif rt and not ra:
            self._label(surf, "Press Space to run A*", x+15, cy, self.font_small, C["text_dim"]); cy += 14

        # Note about what the times mean
        cy += 2
        self._label(surf, "* Pushes = solution quality", x+15, cy, self.font_small, C["text_dim"]); cy += 12
        self._label(surf, "  Compute ms = algo time only", x+15, cy, self.font_small, C["text_dim"]); cy += 12

        cy += 6
        div5 = pygame.Surface((w-30, 1)); div5.fill(C["panel_hi"])
        surf.blit(div5, (x+15, cy)); cy += 8

        # ── LLM Agent section ─────────────────────────────────────────────
        llm_col = C["success"] if "Done" in llm_status else \
                  C["error"]   if "error" in llm_status.lower() or "Error" in llm_status else \
                  C["accent2"] if llm_running else C["text_dim"]
        self._label(surf, "LLM AGENT (G)", x+15, cy, self.font_body, C["text_hi"]); cy += 16
        max_s = (w-30) // 7
        self._label(surf, llm_status[:max_s], x+15, cy, self.font_small, llm_col); cy += 14

        if llm_candidates:
            SC = {"valid": C["success"], "deadlock": C["error"],
                  "illegal": C["text_dim"], "invalid": C["text_dim"]}
            for c in llm_candidates[:3]:
                mv  = c.get("move","?"); st = c.get("status","invalid")
                sc2 = c.get("score",0);  rs = c.get("reasoning","")
                col = SC.get(st, C["text_dim"])
                bt2 = self.font_small.render(f'{mv} {sc2}/10', True, C["bg"])
                bw3 = bt2.get_width()+8; bh3 = bt2.get_height()+4
                draw_rounded_rect(surf, col, pygame.Rect(x+15, cy, bw3, bh3), 3)
                surf.blit(bt2, (x+19, cy+2))
                mc = (w-35-bw3)//7
                self._label(surf, rs[:mc], x+15+bw3+4, cy+3, self.font_small, C["text_dim"])
                cy += bh3+3

        if llm_reasoning:
            words = llm_reasoning.split()
            line_str, lines_out = "", []
            for word in words:
                test = (line_str+" "+word).strip()
                if self.font_small.size(test)[0] < w-35: line_str = test
                else:
                    if line_str: lines_out.append(line_str)
                    line_str = word
            if line_str: lines_out.append(line_str)
            for ln in lines_out[:2]:
                self._label(surf, ln, x+15, cy, self.font_small, C["path"]); cy += 13

        cy += 6
        div_llm = pygame.Surface((w-30, 1)); div_llm.fill(C["panel_hi"])
        surf.blit(div_llm, (x+15, cy)); cy += 8

        # ── A* Playback ──────────────────────────────────────────────────
        self._label(surf, "A* PLAYBACK", x+15, cy, self.font_body, C["text_hi"]); cy += 16
        total = len(sr.full_path) if (sr and sr.solved) else 0
        self._label(surf, f"Step {left.playback_step+1}/{total}", x+15, cy, self.font_title, C["text"]); cy += 20
        self._label(surf, f"Enter Play/Pause  Z/X Speed:{speed}x", x+15, cy, self.font_small, C["path"]); cy += 14
        self._label(surf, f"ToT speed: {tot_speed}x  C/V to change", x+15, cy, self.font_small, C["accent2"]); cy += 16

        div6 = pygame.Surface((w-30, 1)); div6.fill(C["panel_hi"])
        surf.blit(div6, (x+15, cy)); cy += 8

        # ── Controls ─────────────────────────────────────────────────────
        self._label(surf, "CONTROLS", x+15, cy, self.font_body, C["text_hi"]); cy += 14
        for key, desc in [("Space","Run A*"),("T","Heuristic ToT"),("G","LLM ToT"),
                          ("M","Manual / Truth Scanner"),
                          ("Z/X","A* speed"),("C/V","ToT speed"),
                          ("R","Restart"),("O","Overlay"),("N/P","Level"),("L","Select")]:
            ks=self.font_small.render(key, True, C["bg"])
            bw2=ks.get_width()+8; bh2=ks.get_height()+4
            draw_rounded_rect(surf, C["accent"], pygame.Rect(x+15, cy, bw2, bh2), 3)
            surf.blit(ks, (x+19, cy+2))
            self._label(surf, desc, x+15+bw2+5, cy+2, self.font_small, C["text"])
            cy += bh2+3

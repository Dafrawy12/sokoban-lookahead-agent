"""
truth_scanner.py - Symbolic Verification Constraint Demo
=========================================================

This module demonstrates the Physics Verifier acting as a "Truth Scanner"
for the Hybrid ToT agent. It shows the formal mapping:

    Phi: S x A -> S union {null}

where Phi(s, a) = null (rejected) if action a from state s produces:
  - an illegal move   (wall collision, immovable box)
  - a deadlock state  (box irreversibly stuck - corner, edge, etc.)

Run standalone:
    python truth_scanner.py

This produces a terminal report showing the verifier catching each type
of error a proposer (LLM or otherwise) might hallucinate.
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from game  import load_levels, level_to_state, GameState, TARGET
from solver import has_deadlock, deadlock_type

# ANSI colours for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SEP = "-" * 60


def render_board(state: GameState) -> str:
    """ASCII render of board state."""
    rows = []
    for r in range(state.rows):
        row = ""
        for c in range(state.cols):
            cell = state.grid[r][c]
            pos  = (r, c)
            if pos == state.player:
                row += "+" if cell == TARGET else "@"
            elif pos in state.boxes:
                row += "*" if cell == TARGET else "$"
            else:
                row += cell
        rows.append("  " + row)
    return "\n".join(rows)


def phi(state: GameState, action: str) -> tuple:
    """
    Phi(s, a) - the Physics Verifier mapping.

    Returns (next_state, verdict, reason) where:
      verdict = 'ACCEPTED' | 'REJECTED_ILLEGAL' | 'REJECTED_DEADLOCK'
    """
    from game import DIRECTIONS
    if action not in DIRECTIONS:
        return None, "REJECTED_ILLEGAL", "Unknown action - not in action space"

    ns = state.apply_move(action)
    if ns is None:
        return None, "REJECTED_ILLEGAL", "Move blocked by wall or immovable box"

    dl = deadlock_type(ns)
    if dl:
        return None, "REJECTED_DEADLOCK", f"Deadlock detected: {dl}"

    if has_deadlock(ns):
        return None, "REJECTED_DEADLOCK", "Deadlock detected: box irrevocably stuck"

    return ns, "ACCEPTED", "Move is legal and state remains solvable"


def run_scenario(title: str, description: str,
                 state: GameState, proposals: list[tuple[str, str]]):
    """
    Run one verification scenario.
    proposals = list of (action, llm_reasoning) pairs.
    """
    print(f"\n{BOLD}{CYAN}{SEP}{RESET}")
    print(f"{BOLD}SCENARIO: {title}{RESET}")
    print(f"{description}")
    print(f"\nBoard state:")
    print(render_board(state))
    print(f"\n{'Action':<8} {'LLM Reasoning':<42} {'Phi(s,a) Verdict'}")
    print("-" * 90)

    for action, reasoning in proposals:
        ns, verdict, reason = phi(state, action)
        if verdict == "ACCEPTED":
            colour = GREEN
            symbol = "[ACCEPTED]"
        elif verdict == "REJECTED_DEADLOCK":
            colour = RED
            symbol = "[REJECTED]"
        else:
            colour = YELLOW
            symbol = "[REJECTED]"

        print(f"{colour}{symbol} {action:<7}{RESET} "
              f"{reasoning:<42} "
              f"{colour}{verdict}{RESET}")
        print(f"          {'':42} -> {reason}")

    print()


def main():
    levels = load_levels(pathlib.Path(__file__).parent / "levels" / "levels.json")

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  TRUTH SCANNER - Physics Verifier (Phi) Demonstration{RESET}")
    print(f"{BOLD}  Formal mapping: Phi: S x A -> S union {{null}}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print("""
  The LLM proposer (Psi) suggests candidate moves each step.
  The Physics Verifier (Phi) acts as a symbolic constraint,
  rejecting any move that is:
    (1) Physically illegal  - wall or immovable box
    (2) Deadlock-inducing   - box becomes irrevocably stuck

  This is the "Truth Scanner": it catches LLM hallucinations
  before they corrupt the search.
""")

    # -- Scenario 1: Illegal move into a wall ------------------------------
    # Level 1 "Baby Steps":
    #   ######
    #   #    #
    #   # $. #
    #   #  @ #     player at (3,3), box at (2,2), target at (2,3)
    #   ######
    # Move player LEFT twice so they are at (3,1) - one step from the wall
    state1 = level_to_state(levels[0])
    for move in ["LEFT", "LEFT"]:
        ns = state1.apply_move(move)
        if ns: state1 = ns

    run_scenario(
        title="LLM proposes walking into a wall",
        description=(
            "  Level: Baby Steps\n"
            "  Player walked left twice and is now at column 1,\n"
            "  one step from the left wall. LLM hallucinates that\n"
            "  moving LEFT again continues a valid path."
        ),
        state=state1,
        proposals=[
            ("LEFT",  "Continue left - wall not considered    "),   # hits wall
            ("UP",    "Move up to align with box row          "),
            ("RIGHT", "Move right toward box                  "),
        ]
    )

    # -- Scenario 2: LLM proposes pushing box into corner deadlock ---------
    # Level 3 "Corner Work" - find a state where pushing creates a deadlock
    state3 = level_to_state(levels[2])
    # Manually walk to a state where a bad push is possible
    # Apply a sequence that puts us near a corner push opportunity
    test_state = state3
    for move in ["UP", "LEFT", "UP"]:
        ns = test_state.apply_move(move)
        if ns:
            test_state = ns

    run_scenario(
        title="LLM proposes a move that creates a deadlock",
        description=(
            "  Level: Corner Work\n"
            "  LLM suggests pushing the box in a direction that traps\n"
            "  it against a wall with no target - level becomes unsolvable."
        ),
        state=test_state,
        proposals=[
            ("UP",    "Push box up toward target area         "),
            ("DOWN",  "Push box down - LLM ignores corner trap"),
            ("RIGHT", "Reposition player to right side        "),
            ("LEFT",  "Push box left - good directional move  "),
        ]
    )

    # -- Scenario 3: All four moves from a real mid-game state -------------
    state5 = level_to_state(levels[4])   # Cross Roads - harder level
    mid_state = state5
    for move in ["UP", "UP", "RIGHT", "RIGHT", "DOWN"]:
        ns = mid_state.apply_move(move)
        if ns:
            mid_state = ns

    run_scenario(
        title="Full candidate evaluation - all four directions",
        description=(
            "  Level: Cross Roads (mid-game state)\n"
            "  LLM proposes all 4 directions. Verifier filters the set\n"
            "  down to only legally safe moves for the heuristic to score."
        ),
        state=mid_state,
        proposals=[
            ("UP",    "Move player up toward nearest box      "),
            ("DOWN",  "Move player down                       "),
            ("LEFT",  "Move left - possible box push          "),
            ("RIGHT", "Move right                             "),
        ]
    )

    # -- Summary -----------------------------------------------------------
    print(f"{BOLD}{CYAN}{SEP}{RESET}")
    print(f"{BOLD}TRUTH SCANNER SUMMARY{RESET}")
    print(f"""
  The Physics Verifier (Phi) acts as a hard symbolic constraint.
  No hallucinated move can enter the search tree - regardless
  of how confident the LLM proposer sounds.

  This is the Neuro-Symbolic architecture:
    • Psi (LLM)  = fast, flexible, pattern-based proposal
    • Phi (code) = exact, formal, irrefutable verification

  The combination is strictly stronger than either alone.
  Psi without Phi = unconstrained hallucination
  Phi without Psi = blind exhaustive search
  Psi + Phi       = guided search with formal correctness guarantees
""")
    print(f"{BOLD}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()

"""Kitchen layouts (original designs for this project).

two_team_kitchen (main, 4 agents): left-right mirror symmetric; the mirror swaps onion and tomato
dispensers, so an onion team and a tomato team face mirror-image tasks. Shared pots, dish dispensers and
serving windows: teams interact physically but are paid only for their own recipe.

small_kitchen_2p (secondary, 2 agents, one team): standard two-agent cooperative cooking; it cannot test
latent multi-team identification (every pair is a teammate pair).
"""

LAYOUTS = {
    "two_team_kitchen": [
        "XXXPXXXPXXX",
        "O 1     2 T",
        "D         D",
        "S   XXX   S",
        "X         X",
        "T 3     4 O",
        "XXXPXXXPXXX",
    ],
    "small_kitchen_2p": [
        "XXPXPXX",
        "O1   2O",
        "X     X",
        "XDXSXDX",
    ],
}

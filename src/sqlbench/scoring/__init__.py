"""Result normalization and scoring. See docs/scoring.md for the rules.

The rules freeze before the first model run. Any later change to how values
are normalized, matched or classified bumps SCORER_VERSION, gets a CHANGELOG
entry, and requires re-scoring every stored result.
"""

SCORER_VERSION = 1

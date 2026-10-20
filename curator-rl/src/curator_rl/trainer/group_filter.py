"""Question-level filtering (D-102): train only on answer groups that carry a learning signal.

A GRPO group whose G answers all get the same reward has zero advantage everywhere and moves nothing.
In the first 13 GPU runs 35-50% of the groups of every step were like that, whatever the arm mixture
(the scheduler chooses arms; within an arm, questions are all-right or all-wrong this often). Online
difficulty filtering (Bae et al., EACL 2026) and DAPO's dynamic sampling draw more questions than a
step needs and keep the informative groups; GRESO skips them before generation but needs a question's
history from an earlier epoch, which runs shorter than one epoch never have.

Here a step draws ceil(f x P) questions, generates and scores all of them, and trains on P groups:
those with a reward signal first, in the order drawn, then (if too few) the rest. Every generated
token is charged to the run's budget, so a filtered method is compared with an unfiltered one at the
same GPU-hours. Advantages are normalised per group, so keeping whole groups leaves them exact.

A method name ending in "_q" (curator_q, uniform_q, ...) is the base method plus this filtering.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

FILTER_SUFFIX = "_q"
DEFAULT_FACTOR = 1.75     # ~45% of groups carry no signal: 1.75 x P candidates leave ~P informative ones


def split_method(method: str) -> tuple[str, bool]:
    """(scheduler method, question filtering on?): ('curator', True) for 'curator_q'."""
    if method.endswith(FILTER_SUFFIX) and len(method) > len(FILTER_SUFFIX):
        return method[: -len(FILTER_SUFFIX)], True
    return method, False


def candidates(n_keep: int, factor: float) -> int:
    """Questions to draw for a step that trains on n_keep of them."""
    return max(n_keep, math.ceil(n_keep * factor)) if factor > 1.0 else n_keep


def has_signal(values: Sequence[float], tol: float = 1e-6) -> bool:
    """A group's rewards (or advantages) differ: it moves the policy."""
    return bool(values) and max(values) - min(values) > tol


def select_groups(group_has_signal: Sequence[bool], n_keep: int) -> list[int]:
    """Indices of the groups to train on, in their original order: informative ones first, then the rest."""
    signal = [i for i, s in enumerate(group_has_signal) if s]
    rest = [i for i, s in enumerate(group_has_signal) if not s]
    return sorted((signal + rest)[:n_keep])

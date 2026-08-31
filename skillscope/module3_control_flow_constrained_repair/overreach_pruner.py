from __future__ import annotations

from typing import NoReturn

from skillscope.common.models import CandidateAction, UEGNode


class OverreachPruner:
    """Compatibility shim for the retired finite-sample pruning policy."""

    def build_item(
        self,
        *,
        candidate: CandidateAction,
        node: UEGNode | None,
        unnecessary_task_summaries: list[str],
    ) -> NoReturn:
        del candidate, node, unnecessary_task_summaries
        raise RuntimeError(
            "Permanent pruning is disabled: representative tasks are finite, "
            "so Module 3 must synthesize a deny-by-default guard."
        )

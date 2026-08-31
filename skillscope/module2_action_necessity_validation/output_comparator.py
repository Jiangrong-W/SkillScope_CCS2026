from __future__ import annotations

import re


class OutputComparator:
    def compare(
        self,
        *,
        prompt: str,
        original_output: str,
        replay_output: str,
        task_summary: str,
        original_output_grounded: bool = True,
        replay_output_grounded: bool = True,
    ) -> dict[str, object]:
        original_norm = self._normalize(original_output)
        replay_norm = self._normalize(replay_output)
        prompt_norm = self._normalize(prompt)
        task_norm = self._normalize(task_summary)

        uncertainty_flags: list[str] = []
        if not original_norm:
            uncertainty_flags.append("missing_original_output_evidence")
        if not replay_norm:
            uncertainty_flags.append("missing_replay_output_evidence")
        if not original_output_grounded:
            uncertainty_flags.append("ungrounded_original_final_output")
        if not replay_output_grounded:
            uncertainty_flags.append("ungrounded_replay_final_output")
        evidence_sufficient = bool(
            original_norm
            and replay_norm
            and original_output_grounded
            and replay_output_grounded
        )
        if not evidence_sufficient:
            uncertainty_flags.append("material_goal_ambiguity")

        equivalent = evidence_sufficient and original_norm == replay_norm
        if evidence_sufficient and not equivalent and task_norm:
            equivalent = task_norm in original_norm and task_norm in replay_norm
        if evidence_sufficient and not equivalent and prompt_norm:
            prompt_tokens = set(prompt_norm.split())
            original_tokens = set(original_norm.split())
            replay_tokens = set(replay_norm.split())
            if prompt_tokens:
                overlap_original = len(prompt_tokens & original_tokens) / len(prompt_tokens)
                overlap_replay = len(prompt_tokens & replay_tokens) / len(prompt_tokens)
                equivalent = overlap_original >= 0.4 and overlap_replay >= 0.4

        if not evidence_sufficient:
            reason = (
                "The original/replay outputs do not provide enough observable "
                "evidence to establish GoalSat."
            )
        elif equivalent:
            reason = (
                "Original and replay outputs remain semantically aligned with "
                "the task request."
            )
        else:
            reason = (
                "Replay output no longer looks equivalent to the original task "
                "outcome."
            )
        return {
            "equivalent": equivalent,
            "evidence_sufficient": evidence_sufficient,
            "reason": reason,
            "strategy": "grounded_task_output_comparison",
            "original_output_grounded": original_output_grounded,
            "replay_output_grounded": replay_output_grounded,
            "uncertainty_flags": uncertainty_flags,
        }

    def _normalize(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        normalized = re.sub(r"[^a-z0-9 _.-]", "", normalized)
        return normalized

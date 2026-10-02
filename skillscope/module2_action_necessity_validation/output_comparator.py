from __future__ import annotations

import re
import unicodedata


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
        replay_result_absence_observed: bool = False,
    ) -> dict[str, object]:
        original_norm = self._normalize(original_output)
        replay_norm = self._normalize(replay_output)

        # A completed, observed run with no result on a required text-output
        # channel is a negative outcome, rather than missing telemetry. The
        # caller must establish capture integrity before enabling this branch.
        if (
            original_norm
            and original_output_grounded
            and replay_result_absence_observed
            and self._requires_text_result(prompt)
        ):
            return {
                "equivalent": False,
                "evidence_sufficient": True,
                "reason": "The original produced the requested text result; the completed replay observably produced no result on that channel.",
                "strategy": "observed_required_text_result_loss",
                "original_output_grounded": True,
                "replay_output_grounded": replay_output_grounded,
                "replay_result_absence_observed": True,
                "uncertainty_flags": [],
            }

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

        if not evidence_sufficient:
            reason = (
                "The original/replay outputs do not provide enough observable "
                "evidence to establish GoalSat."
            )
        elif equivalent:
            reason = (
                "The grounded outputs match after Unicode canonicalization "
                "and line-ending normalization."
            )
        else:
            reason = (
                "The grounded outputs differ. Lexical overlap or a shared "
                "task description cannot establish result equivalence; a "
                "separate semantic judgment is required for paraphrases."
            )
        return {
            "equivalent": equivalent,
            "evidence_sufficient": evidence_sufficient,
            "reason": reason,
            "strategy": "grounded_exact_text_comparison",
            "original_output_grounded": original_output_grounded,
            "replay_output_grounded": replay_output_grounded,
            "replay_result_absence_observed": replay_result_absence_observed,
            "uncertainty_flags": uncertainty_flags,
        }

    def _normalize(self, text: str) -> str:
        # Preserve whitespace as well as punctuation and case. Formatting,
        # quoted strings and code indentation can be the requested result.
        # Canonically equivalent Unicode and platform line endings are the
        # only equivalences this task-independent comparator establishes.
        if not text.strip():
            return ""
        return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))

    @staticmethod
    def _requires_text_result(prompt: str) -> bool:
        """Recognize text-result tasks without assuming file tasks need stdout."""
        text = prompt.casefold()
        if re.search(r"\b(?:stdout|print|answer|respond|reply)\b|打印|回答", text):
            return True
        artifact_goal = re.search(
            r"\b(?:save|write|export|create|generate)\b.{0,60}\b(?:file|pdf|docx|xlsx|pptx|artifact)\b|保存.{0,20}文件|生成.{0,20}文件",
            text,
        )
        return not artifact_goal and bool(
            re.search(r"\b(?:summarize|summary|calculate|format|return|show|tell)\b|摘要|总结|计算|返回|显示", text)
        )

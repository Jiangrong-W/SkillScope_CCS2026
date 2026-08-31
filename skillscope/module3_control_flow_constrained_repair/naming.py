from __future__ import annotations


def overreach_id_from_candidate_id(candidate_id: str) -> str:
    if candidate_id.startswith("candidate-"):
        return "overreach-" + candidate_id[len("candidate-") :]
    if candidate_id.startswith("candidate_"):
        return "overreach_" + candidate_id[len("candidate_") :]
    return f"overreach-{candidate_id}"

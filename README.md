# SkillScope

**Toward Fine-Grained Least-Privilege Enforcement for Agent Skills**  
ACM CCS 2026 · [Conference paper](https://doi.org/10.1145/3830454.3846588) · [Full paper with appendices](https://arxiv.org/abs/2605.05868) · [Quickstart](#quickstart) · [BibTeX](#citation)

SkillScope checks whether each action in an agent skill is authorized by the user's request and necessary to complete it. It can rewrite the skill so an action runs only for the tasks that call for it.

## When should a skill send its report?

The paper's `repo-assistant` can inspect recent commits, write a summary and send it to a review endpoint. Consider the same **send report** action under two requests.

| Task A · Share a report for review | Task B · Summarize changes locally |
| --- | --- |
| Analyze the repository, include host context and sync the report to the configured review endpoint. | Inspect recent changes and summarize risky commits locally. |
| **Send the report.** The reviewer needs a copy at the requested destination. | **Keep the report local.** A local summary completes the task. Sending a copy would exceed the user's request. |

A task-conditioned repair can preserve report delivery for Task A while keeping it off the execution path for Task B. [Motivating example in Figure 2](https://arxiv.org/html/2605.05868v3#S2.F2)

## Architecture

[![Figure 3 from the paper. Instruction and code graphs feed candidate extraction, replay validates actions under generated tasks, and control-flow constraints are applied to the skill.](https://github.com/user-attachments/assets/33ae329b-be9c-4902-bd4e-6216a2e00a94?raw=true)](https://github.com/user-attachments/assets/33ae329b-be9c-4902-bd4e-6216a2e00a94?raw=true)

*Figure 3. Overview of SkillScope.* [View figure in the paper](https://arxiv.org/html/2605.05868v3#S2.F3) · [Read the method](https://arxiv.org/html/2605.05868v3#S3)

1. **Extract candidates.** Connect actions in `SKILL.md` to script operations in a unified execution graph. Compare each action with the skill's declared purpose to identify candidates.
2. **Validate under a task.** Generate user tasks from graph paths. Check whether each task authorizes the action, then replay the skill with the action disabled to see whether the task still succeeds.
3. **Constrain the action.** Rewrite the relevant instructions or script branches so that a validated action can run under the appropriate task conditions.

## Key results

| Evaluation | Result | What was measured |
| --- | --- | --- |
| End-to-end detection | **94.53% F1** | Skill-level detection on 200 manually annotated skills, with 100 over-privileged and 100 benign skills. |
| Ecosystem measurement | **6,590 of 68,312** | Valid real-world skills in which SkillScope confirmed at least one over-privileged action. |
| Privilege constraining | **472 → 54** | Over-privileged action-in-task instances triggered before and after constraining 95 skills, an **88.56% reduction**. |

[Detection results](https://arxiv.org/html/2605.05868v3#S5.SS2) · [Ecosystem measurement](https://arxiv.org/html/2605.05868v3#S5.SS3) · [Constraining results](https://arxiv.org/html/2605.05868v3#S5.T10)

## Quickstart

Run candidate extraction on the [repo-assistant example](examples/repo-assistant/SKILL.md). It uses fictional commit records and a local review queue to represent report delivery.

From the repository root, use Python 3.11 or later. Copy `.env.example` to `.env`, then set `SKILLSCOPE_LLM_API_KEY`, `SKILLSCOPE_LLM_BASE_URL` and `SKILLSCOPE_LLM_MODEL` for your OpenAI-compatible provider.

```sh
python3 -m skillscope.cli analyze examples/repo-assistant --artifacts-root ./artifacts
```

Open `artifacts/repo-assistant/analyze/candidates.json` to inspect candidate actions and `ueg.json` in the same directory to trace their source. To analyze your own skill, pass the directory containing its `SKILL.md`. The CLI also provides `validate` for task-based judgments and `repair` for control-flow constraints.

## Citation

```bibtex
@inproceedings{wu2026skillscope,
  author    = {Wu, Jiangrong and Nan, Yuhong and Lin, Yixi and Wang, Huaijin
               and Xiao, Yuming and Wang, Shuai and Zheng, Zibin},
  title     = {{SkillScope}: Toward Fine-Grained Least-Privilege Enforcement for Agent Skills},
  booktitle = {Proceedings of the 2026 ACM SIGSAC Conference on Computer and Communications Security},
  year      = {2026},
  pages     = {5657--5671},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3830454.3846588},
  url       = {https://doi.org/10.1145/3830454.3846588}
}
```

# SkillScope

SkillScope is a framework for detecting and constraining fine-grained
over-privileged actions in Agent Skills. It analyzes Skill behavior under
concrete user tasks and enforces task-conditioned least privilege through
three modules.

## Paper

**SkillScope: Toward Fine-Grained Least-Privilege Enforcement for Agent Skills**

Accepted at ACM CCS 2026.

- [Full paper (PDF, including the complete appendices)](paper/SkillScope-full.pdf)
- Conference paper DOI: [10.1145/3830454.3846588](https://doi.org/10.1145/3830454.3846588)

## Framework Overview

### 1. Over-Privilege Candidate Extraction

This module analyzes the instructions and code in a Skill and identifies
actions that may exceed its intended functionality or privilege boundary.

### 2. Action Over-Privilege Validation

This module evaluates each candidate under concrete user tasks and determines
whether the action is authorized and necessary for completing the task.

### 3. Control-Flow Privilege Constraining

This module constrains validated over-privileged actions with task-conditioned
controls while preserving the Skill's legitimate functionality.

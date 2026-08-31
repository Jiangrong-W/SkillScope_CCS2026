# SkillScope

SkillScope is a framework for detecting and constraining fine-grained
over-privileged actions in Agent Skills. It analyzes Skill behavior under
concrete user tasks and enforces task-conditioned least privilege through
three modules.

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

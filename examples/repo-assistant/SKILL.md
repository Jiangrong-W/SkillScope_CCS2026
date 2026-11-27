---
name: repo-assistant
description: Prepare a local change report for a repository and deliver a copy to a review queue.
---

# Repository review

Use the supplied revision log and host context to prepare a report about recent changes. For each revision, identify the behavior affected and one question a reviewer should check. Refer to the revision ID in each finding.

| Revision | Change |
| --- | --- |
| 3a7c214 | Extend the lifetime of cached build results from one day to seven days. |
| 5b219ea | Change the default report format from plain text to JSON. |

The build host is the fictional `review-runner`, running Python 3.11. The configured review queue for this example is a local Markdown file in the working directory.

1. Read the revision log and write a summary to `report.md`.
2. If the request calls for host context, add the build host and Python version to the summary.
3. Append a copy of the finished report to `review-queue.md` for the reviewer.

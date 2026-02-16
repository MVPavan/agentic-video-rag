# AGENTS.md

## Purpose
This file defines how coding agents must operate in this repository to build the Agentic Video RAG system safely, consistently, and with high signal.

## Scope and Priority
1. This file applies to the entire repository.
2. If a deeper `AGENTS.md` is added in a subdirectory, it overrides this file for that subtree.
3. Product truth source: `design/spec_groundtruth.md`.
4. Execution tracking source: `design/execution_plan.md` (live milestone and status file).
5. If code and spec disagree, update code to match the spec or propose a spec change explicitly.

## Codex Instruction Discovery
Codex should be expected to discover instructions from multiple layers:
1. `~/.codex/AGENTS.md` (user/global defaults).
2. Repository root `AGENTS.md` (this file).
3. Subdirectory `AGENTS.md` files (more specific scope).

Priority rule: the most specific in-scope file wins if there is a conflict.

## Local Overrides (Do Not Commit)
For personal/local machine customization, use `AGENTS.override.md`.
1. `AGENTS.override.md` may override this file for local workflows.
2. `AGENTS.override.md` must stay uncommitted (add to `.gitignore` if needed).
3. Project-wide policy changes must go into `AGENTS.md`, not override files.

## Project North Star
Ship an evidence-grounded 7-stage Video RAG pipeline that:
1. Produces camera/time-linked claims.
2. Preserves entity consistency across cameras.
3. Surfaces uncertainty instead of hallucinating certainty.
4. Is reproducible via strict config and versioned artifacts.

## Non-Negotiable Engineering Rules
1. Config format must be YAML.
2. Config composition/merging must use OmegaConf.
3. All merged config must be validated with strict Pydantic (`extra="forbid"`).
4. Follow DRY strictly: define model/store IDs and thresholds once; reference everywhere else.
5. Never hardcode thresholds/model IDs in runtime logic when they belong in config.
6. Every claim-producing path must preserve evidence references.
7. Any fallback path must emit explicit uncertainty/failure flags.

## Architecture Guardrails
1. Preserve the 7-stage contract from `design/spec_groundtruth.md`.
2. Keep stage boundaries explicit: ingestion, retrieval, grounding, ReID, temporal localization, graph memory, synthesis.
3. Use unresolved states for ambiguous identity links; do not force merges.
4. Retrieval confidence is not proof; temporal grounding + evidence linking are required before synthesis.

## Required Deliverables Per Change
For any non-trivial change, include:
1. Code changes.
2. Config/schema updates (if behavior changes).
3. Tests for the changed behavior.
4. Execution progress update in `design/execution_plan.md` when milestone status changes.
5. Short note in the Learning Log (see "Continuous Improvement Rules").

## Instruction Quality Rules (Best Practices)
When editing this file, keep instructions high signal:
1. Be explicit, concrete, and testable.
2. Prefer repo-specific rules over generic advice.
3. State priorities and exceptions clearly.
4. Avoid contradictory requirements across sections.
5. Keep this file concise; move long procedures to dedicated docs and link them.

## Repository Conventions
1. Use `src/` for implementation code.
2. Use `tests/` for tests.
3. Use `config/` for YAML configs and override profiles.
4. Use `scripts/` for runnable helpers.
5. Keep experimental notebooks and one-off scripts out of core runtime paths.

## Coding Standards
1. Python 3.11+ only.
2. Type hints required on public functions.
3. Prefer small, pure functions for stage logic.
4. Avoid hidden global state.
5. Use clear names matching spec terms (`stage_1`, `activity_ingestion`, `ObjectClusterID`, etc.).
6. Raise explicit errors for broken contracts; fail early.

## Config and Schema Standards
1. Centralize constants (thresholds, top-k, retry limits).
2. Add cross-reference validators (stage IDs, model IDs, datastore/resource IDs).
3. Maintain a single `stage_catalog` map from `stage_id` to crisp `stage_name`.
4. Validate stage completeness (Stage 1..7 exactly once in stage specs).
5. Validate each stage's `stage_name` against the canonical `stage_catalog` mapping.
6. Constrain confidence-like parameters to `[0, 1]`.
7. Add migration notes when renaming config keys.

## Testing Standards
Minimum required test coverage for each new feature:
1. Happy-path unit test.
2. At least one failure/edge-case test.
3. Config validation test for relevant schema changes.
4. Regression test when fixing a bug.

Pipeline-specific checks should include:
1. Stage I/O contract validation.
2. Evidence linkage completeness for synthesizeable claims.
3. Ambiguity handling (unresolved identities remain unresolved).

## Agent Execution Workflow
1. Read relevant spec section(s) first.
2. Read and align with current milestone state in `design/execution_plan.md`.
3. State assumptions briefly before major edits.
4. Implement smallest coherent change that passes tests.
5. Run tests/lint for touched components.
6. Update docs/config/tests together.
7. Update `design/execution_plan.md` when status, risk, or dates changed.
8. Add a Learning Log entry for substantial changes.

## Execution Plan Operating Rules
Treat `design/execution_plan.md` as a live control file.
1. Keep milestone IDs stable; do not rename without explicit migration note.
2. Allowed statuses: `not_started`, `in_progress`, `blocked`, `done`.
3. On status transition to `in_progress`, set `Start Date` if empty.
4. On status transition to `done`, set `Completed Date`.
5. If `blocked`, include clear unblock condition in `Notes / Risks`.
6. Do not mark a milestone `done` unless its acceptance gate is satisfied.
7. If a change affects scope/timeline, update `Target Date` and note reason.

## Standard Commands (Update As Repo Evolves)
Agents should prefer these canonical commands when available:
1. Setup: `uv sync` (or project-approved equivalent).
2. Tests: `uv run pytest`.
3. Lint: `uv run ruff check .`.
4. Format: `uv run ruff format .`.

If these commands change, update this section in the same PR.

## What Agents Must Avoid
1. Bypassing schema validation for speed.
2. Introducing duplicate config values across files.
3. Coupling stage internals tightly across boundaries.
4. Silent fallback behavior without logging/flags.
5. Large refactors without incremental verification.

## Pull Request / Change Checklist
Before finalizing, agents must verify:
1. Behavior matches `design/spec_groundtruth.md`.
2. `design/execution_plan.md` status/notes are updated for impacted milestone(s).
3. YAML + OmegaConf + Pydantic flow is preserved.
4. DRY constraints are upheld (no duplicated constants/IDs).
5. Tests cover new behavior and pass.
6. Learning Log updated when applicable.

## Continuous Improvement Rules (Living AGENTS.md)
This file must evolve with real project outcomes.

### When to Update
Update this file when any of the following occurs:
1. A repeated failure pattern is discovered.
2. A new practice significantly improves delivery speed or quality.
3. A rule here is found ambiguous, outdated, or counterproductive.
4. A new subsystem or stage-level constraint is introduced.

### How to Update
1. Keep updates small and specific.
2. Prefer adding concrete rules over broad statements.
3. Record the reason in the Learning Log table.
4. If a rule changes behavior, include effective date.
5. Remove or rewrite stale rules that no longer reflect real workflows.
6. If file length grows too much, split operational detail into linked docs while keeping this file as the concise control plane.

### Learning Log (What Worked / What Didn't)
Add entries in reverse chronological order.

| Date | Area | What Worked | What Didn't | Action / Rule Update |
|---|---|---|---|---|
| 2026-02-16 | Onboarding clarity | A single root README with commands, stage map, and extension workflow reduces startup friction across new chats/agents. | Spreading onboarding details only across spec and plan docs slows ramp-up. | Keep `README.md` as the practical entrypoint and update it when run/test commands or module anchors change. |
| 2026-02-16 | Full-pipeline delivery | Implementing all stage contracts with deterministic adapters enabled complete P2-P8 validation in one passable runtime. | Waiting for real model integrations before contract-level tests would have blocked milestone progress. | Keep a deterministic reference path that must pass before model-backed integrations. |
| 2026-02-16 | Retrieval robustness | Combining full-query scoring with decomposed-query recall and clip diversity improved downstream grounding/graph quality. | Single-query ranking could miss critical windows, causing empty evidence graphs. | Enforce decomposed-query recall path and clip-diversity selection in Stage 2. |
| 2026-02-16 | P1 foundation | Building strict schema + merge loader first made later stage code safer and easier to test. | Starting orchestration runtime wiring before contracts increases rework risk. | Keep milestone order: config/schema (`M1.1`/`M1.2`) before full orchestration runtime wiring (`M1.3`). |
| 2026-02-16 | Tracking hygiene | Separating stable spec and live execution tracker reduces spec churn and review noise. | Keeping milestones in the SSOT spec mixed stable contracts with rapidly changing status data. | Set `design/execution_plan.md` as live tracking source and made updates mandatory in workflow/checklist. |
| 2026-02-16 | Stage naming | Explicit `stage_id -> stage_name` mapping improves readability and validation. | ID-only stage references are harder to scan and easier to misuse. | Added mandatory `stage_catalog` and stage-name validation rule. |
| 2026-02-16 | Spec governance | Ground-truth spec in Markdown with explicit config governance section reduced ambiguity. | Treating YAML as the spec artifact caused expectation mismatch. | Clarified: Markdown is SSOT; YAML is runtime config only. |

## Decision Log Template (Use in PRs and major commits)
Use this lightweight template in PR description or commit message for meaningful decisions:

1. Context
2. Decision
3. Alternatives considered
4. Trade-offs
5. Follow-up actions

## Escalation Guidelines
Agents should pause and ask for direction when:
1. Spec and user instruction conflict materially.
2. A change requires destructive operations.
3. Required dependencies/tools cannot run in the environment.
4. A stage contract must be broken to proceed.

## Done Criteria
A task is done only when:
1. Implementation is complete.
2. Relevant tests pass.
3. Documentation is updated.
4. Impacted milestone status in `design/execution_plan.md` is updated.
5. Config/schema remains strict and DRY.
6. Evidence/uncertainty behavior is preserved for claim-producing paths.

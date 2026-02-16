# Rollback Notes

## Rollback Strategy
1. Roll back config and runtime code together to last green test commit.
2. Re-run `scripts/validate_runtime_config.py` and full unit tests.
3. Re-open milestone status in `design/execution_plan.md` as `in_progress` or `blocked`.

## High-Risk Areas
1. Stage catalog mapping and schema validation logic.
2. Stage 6 evidence-linking contracts.
3. Stage 7 redaction behavior for unverifiable claims.

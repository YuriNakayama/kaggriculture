---
description: Check CI pipeline status and identify/fix failures when detected
---

# CI Fix

Check the status of backend and frontend CI pipelines, identify the root cause of failures, and apply fixes.

## Steps

### 1. Check current branch and PR

```bash
git branch --show-current
gh pr view --json number,title,state,statusCheckRollup
```

### 2. Check CI workflow run status

```bash
# Check latest CI runs
gh run list --workflow=ci-backend.yml --limit=3
gh run list --workflow=ci-frontend.yml --limit=3
```

### 3. If failing, check logs

```bash
# Get detailed logs for the failed run
gh run view <run-id> --log-failed
```

### 4. Analyze and fix failure cause

#### Backend CI failure patterns

| Step | Check command | Fix method |
|------|--------------|------------|
| ruff format | `dev/test` (includes format check) | `dev/format` |
| ruff check | `dev/lint` | `dev/lint` (with --fix) |
| mypy | `dev/lint` (includes mypy) | Fix type errors |
| pytest | `dev/test` | Fix test failures |

#### Frontend CI failure patterns

| Step | Check command | Fix method |
|------|--------------|------------|
| lint | `dev/lint` | Fix lint errors |

### 5. Local verification after fix

```bash
# Backend (format check → lint → type check → pytest)
dev/test

# Frontend (type-check → format:check → lint → Mockoon + Playwright)
```

### 6. Commit and push fix to re-run CI

After user confirmation, commit and push the fix.

## Notes

- Always identify the failure cause from logs before attempting a fix
- Reproduce and verify locally before pushing fixes
- When multiple failures exist, address them in priority order (format → lint → type → test)

## Language

- **All user-facing output, reports, and summaries must be written in Japanese(すべてのユーザーへの出力は日本語にしてください)**

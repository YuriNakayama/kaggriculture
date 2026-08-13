---
name: submit
description: >
  Kaggle submission runner for Kaggriculture agent cases. Packages a
  backend/pipeline/<family>/case<N> directory, dry-run verifies it under the
  measured harness conditions (Python 3.11, exec'd main.py, isolated env), and
  submits it to the competition via dev/submit, then monitors the validation
  episode and retrieves agent logs. Use whenever the user types /submit or asks
  to submit a case to Kaggle — "case2 を提出して", "submit rulebase/case3",
  "サブミットして", "リーダーボードに出して". Real submissions consume one of
  5 daily slots, so the skill always dry-runs first and asks for explicit
  confirmation before uploading.
---

# Kaggle Submit

Drive one submission of an agent case to the Kaggriculture competition,
end-to-end. Full background: `.claude/rules/backend/submit.md`.

## Inputs

- **case** (required): `<family>/case<N>`, e.g. `rulebase/case2`. If the user
  did not name one, list `backend/pipeline/*/case*/` and ask with
  `AskUserQuestion`.
- **message** (required for a real submission): short description linking the
  submission to its hypothesis. Default to the case README's one-liner plus
  the current git sha if the user gave none.

## Procedure

1. **Preflight** (no slot consumed):
   - `dev/kaggle submissions kaggriculture` — confirm auth works and see
     today's Kaggle-side history.
   - Count today's local records in `data/output/submit/` — if 5/5 used, stop
     and report; `--force` only on explicit user request.
   - `git status` — if the tree is dirty, tell the user the audit sha will not
     reproduce the archive and recommend committing first (proceed only if
     they accept).

2. **Dry run** (no slot consumed, mandatory — never skip):

   ```bash
   dev/submit --case <family>/caseN --dry-run
   ```

   This verifies under the measured production conditions (Python 3.11,
   `exec`'d `main.py`, isolated interpreter, cwd ≠ agent dir, full 720-turn
   season vs `starter`). On failure, report the error and stop — do not
   attempt a real submission.

3. **Confirm with the user** before spending the slot: show archive size,
   member list, dry-run score, slots used today (n/5). Use `AskUserQuestion`.
   Never submit without this explicit confirmation, even in autonomous mode.

4. **Submit**:

   ```bash
   dev/submit --case <family>/caseN -m "<message>"
   ```

   Confirm the record file appeared under `data/output/submit/`.

5. **Monitor** (background poll, ~60s interval until it leaves PENDING):

   ```bash
   dev/kaggle submissions kaggriculture          # status
   dev/kaggle episodes <SUBMISSION_ID>           # validation episode id
   dev/kaggle logs <EPISODE_ID> 0                # agent stdout/stderr
   ```

   Logs download into `backend/` — move them to the scratchpad or
   `data/lake/kaggle_episodes/`, never leave them in `backend/`.

6. **Report**: final status (COMPLETE / ERROR), validation score, episode id,
   and — if the validation episode failed — the traceback extracted from the
   agent logs.

## Failure handling

- Dry-run failure → fix or report; never `--skip-verify` to work around it.
- `SubmissionStatus.ERROR` → fetch episode logs, extract the traceback, and
  map it against the measured constraints table in
  `.claude/rules/backend/submit.md` (3.11 syntax? relative import in main.py?
  `__file__` in main.py? bare relative `open()`?).
- Quota refusal → report count and stop; suggest local evaluation
  (`dev/simulate`) instead.

---
name: git-sync
description: >
  Commit and push tracked file changes to GitHub on demand. Use this skill
  whenever the user says anything like "commit my changes", "push to GitHub",
  "git sync", "save my work to GitHub", "/git-sync", or asks to stage/commit/push
  the current state of the repo. Also trigger when the user says "checkpoint",
  "save progress", or "ship it". Generates a structured conventional commit
  message automatically from the diff — user does not need to provide one.
---

# git-sync Skill

Commit all tracked changes and push to the current branch. Invoked on demand.

## Behavior

1. **Check status** — run `git status --short` to confirm there are tracked changes. If nothing to commit, report that and stop.
2. **Show the diff summary** — run `git diff --stat HEAD` so the user can see what's being committed.
3. **Generate commit message** — inspect `git diff HEAD` (or `git diff --cached` after staging) and write a conventional commit message:
   - Format: `<type>(<scope>): <short imperative description>`
   - Types to use for this project:
     - `feat` — new feature or pipeline component
     - `fix` — bug fix
     - `data` — data ingestion, scraping, or schema changes
     - `model` — model training, evaluation, or parameter changes
     - `refactor` — restructuring without behavior change
     - `test` — adding or updating tests
     - `docs` — documentation or CLAUDE.md updates
     - `chore` — deps, config, tooling, CI
   - Scope should be the module or subsystem (e.g., `poisson`, `kalshi`, `features`, `elo`, `pipeline`)
   - Keep subject line under 72 characters
   - Add a short body (2–4 bullet points) if the diff touches multiple areas or the change is non-obvious
4. **Stage tracked files** — run `git add -u` (tracked files only, never `git add -A`)
5. **Commit** — run `git commit -m "<generated message>"`
6. **Push** — run `git push` to the current remote/branch, no branch safety checks
7. **Report** — show the commit hash and summary line; confirm push succeeded

## Rules

- Always use `git add -u`, never `git add -A` — untracked files are never staged
- Never ask the user to provide the commit message; generate it from the diff
- Never rebase, amend, or force-push unless the user explicitly asks
- If `git push` fails (e.g., remote rejected, no upstream set), show the error clearly and suggest the fix (e.g., `git push --set-upstream origin <branch>`) but do not run it automatically
- If there are merge conflicts, stop and report — do not attempt to resolve them

## Commit message examples

```
feat(kalshi): add WebSocket reconnection with exponential backoff

- Reconnects on disconnect with jitter-based backoff (max 60s)
- Tracks subscription state to re-subscribe after reconnect
- Adds unit tests for backoff calculation
```

```
data(pybaseball): cache game logs to SQLite with WAL mode
```

```
fix(poisson): correct lambda shift to prevent data leakage

Rolling window now uses .shift(1) before .rolling() on all
lagged run-rate features per architecture constraint.
```

```
chore: add Ruff config and pre-commit hook
```

## Invocation

User says any of: `/git-sync`, "commit my changes", "push to GitHub", "save progress", "checkpoint", "ship it", or similar.

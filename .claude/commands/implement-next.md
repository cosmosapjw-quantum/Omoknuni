# /implement-next — Implement the next READY task safely

**Goal**  
Pick the next task in `specs/001-goal-create-spec/tasks.md` (topmost not done), implement it *safely*, and update task status + artifacts. Respect SDD: the spec and plan are source of truth.

**Inputs**
- `specs/001-goal-create-spec/spec.md`
- `specs/001-goal-create-spec/plan.md`
- `specs/001-goal-create-spec/tasks.md`

**Rules**
- Never overwrite existing code without explaining the change and creating a backup (`.bak` or git commit).
- Keep changes scoped to ONE task (≤4h).
- Generate unit tests and minimal docs alongside code.
- Add/adjust CI as needed to keep the build green.
- If the task is ambiguous, add an “Assumption:” note in the PR description and the “Open Questions” section of `spec.md`.

**Process**
1) **Select Task:** Choose the first task in tasks.md that is not marked complete, starting in Phase 0 and moving downward.
2) **Plan:** Summarize the steps you will take (2–6 bullets). Confirm file paths.
3) **Implement:** Create/update only the files needed. Add unit tests. Keep code idiomatic and minimal.
4) **Validate:** Run or simulate tests. If not possible, add a `HOWTO-RUN-TESTS` block with exact commands.
5) **Document:** Update `tasks.md` acceptance with checkmarks; add a short changelog entry in `CHANGELOG.md` (create if missing).
6) **Propose Next:** Suggest 1–2 follow-up tasks that logically unblock the next phase.

**Outputs**
- Modified code/tests/docs.
- Updated `specs/001-goal-create-spec/tasks.md` for the implemented task:
  - Add a ✅ to the Acceptance line(s) that pass.
  - Timestamp, author, and commit hash (if available).

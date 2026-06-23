---
name: ghs
description: Golden Hoop Spell (ghs) orchestration discipline. Use when the ghs plugin is active (any ghs-* tool has been or should be called). Enforces the init → plan → sprint → code → status → archive workflow order, drives the right-side TODO panel via todowrite at every stage transition, and mandates executing the ▶ NEXT ACTION anchor at the end of each ghs tool response rather than skipping ahead.
---

# ghs Orchestration Skill

This skill guides the main AI through the Golden Hoop Spell (ghs) structured
delivery workflow. It is loaded into the system prompt so the discipline
below is always in effect once ghs is active in a session.

## Canonical Workflow Order

Drive the project through these tools **in this order**; do not invoke a later
stage's tool before the earlier one has completed:

1. `ghs-init` — bootstrap `.ghs/features.json`, `.ghs/progress.md`,
   `.ghs/ghs.json`, and the plan-dispatcher subagent markdowns.
2. `ghs-config` — re-render the 3 subagent markdowns after editing model IDs
   in `.ghs/ghs.json`.
3. `ghs-plan-start` → `ghs-plan-review` → `ghs-plan-finalize` — the 3-role
   plan dispatcher (context snapshot → design → review → finalize). These
   three are a single logical phase; do not interleave other ghs stages
   while a plan is mid-flight.
4. `ghs-sprint` — decompose the finalized plan into atomic features
   (appended to `.ghs/features.json`).
5. `ghs-code` — implement ONE feature per session (or a conflict-free batch
   in parallel mode).
6. `ghs-status` — read-only progress check at any time.
7. `ghs-archive` / `ghs-force-archive` — archive completed sprints.

`ghs-status` is safe to call at any point; every other tool belongs to a
specific stage and its output names the next tool to call.

## Todo Discipline (mechanism one)

The right-side TODO panel is the only durable view of workflow progress, and
the built-in `todowrite` tool is the **only** thing that can render to it.

- **On entering any ghs multi-step workflow** (plan / sprint / code), call
  `todowrite` to build a stage checklist with the current stage marked
  `in_progress`.
- **On every stage transition**, call `todowrite` again: mark the prior stage
  `completed` and the new current stage `in_progress`. A stage transition is
  signalled by a new `ghs stage:` banner in the tool response.
- If a ghs tool response contains a `TODO:` directive, follow it — the
  disconnect-detection state machine observed that the panel was never seeded.
- If a ghs tool response contains a `STALE TODO:` warning, the stage advanced
  but the panel was not refreshed. Call `todowrite` immediately to realign.

Keeping the panel accurate is what lets the disconnect-detection state machine
observe progress; skipping `todowrite` makes mechanism one blind.

## ▶ NEXT ACTION Anchoring

Every ghs multi-step tool response ends with a `▶ NEXT ACTION: <tool call>`
anchor. This anchor is **mandatory**: execute the named tool call exactly as
written. Do NOT:

- skip past it and take over the next step yourself,
- substitute a different tool,
- batch multiple stages into one turn.

If the anchor names a subagent dispatch (e.g. a Task tool call to
`ghs-context-haiku`), perform that dispatch and feed its output back into the
named next ghs tool.

## Broken-Flow Recovery

If you are unsure where the workflow stands (interrupted session, lost
context, or a tool response you cannot reconcile):

1. Call `ghs-status` — it reports the per-sprint feature counts, the
   in-progress feature, the next ready feature, and recent `progress.md`
   entries. This is the single source of truth for "what is the current
   stage".
2. Read `.ghs/progress.md` (most recent session first) for the prior
   session's explicit next-step note.
3. Read `.ghs/features.json` to confirm feature statuses and dependency
   readiness before resuming `ghs-code`.
4. Resume from the stage `ghs-status` indicates — re-seed the `todowrite`
   checklist for that stage before continuing.

Never guess the stage; never restart the workflow from `ghs-init` on an
already-initialised project (it will refuse without `force: true`).

## Reading List (when a stage is unfamiliar)

- `shared/references/coding-agent.md` — the single-feature and parallel-mode
  implementation protocol that `ghs-code` dispatches against.
- `shared/references/plan-designer.md` — how the plan dispatcher's designer
  role should structure a plan, including the optional built-in-plan-agent
  backend.
- `.ghs/features.json` — feature ids, acceptance criteria, dependencies, and
  `files_affected` for every sprint.

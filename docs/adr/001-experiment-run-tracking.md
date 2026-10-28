# ADR-001: Track results by experiment run

**Status:** Accepted  
**Date:** 2026-10-28  
**Decider:** Steven Yabann  

## Context

HydroFed currently stores plant metrics and federation-round metrics as separate
rows in SQLite. Rows identify the experiment type, such as `baseline` or
`federated`, but they do not identify the specific execution that produced them.

This lets the dashboard accidentally combine the newest result from different
runs. That would make comparisons between baseline and federated experiments
unreliable.

## Decision

Add an `experiment_runs` table. Each new baseline or federated execution will
receive a UUID `run_id`. All result rows generated during that execution will
store the same `run_id`.

Each run stores:

- experiment type
- status
- configuration metadata
- start and finish times
- optional failure message

Existing result rows will remain unassigned instead of being guessed into runs.

## Options considered

### Group rows by timestamp

This is simple but unreliable because overlapping runs or delayed writes can
produce ambiguous groups.

### Store only a timestamp in each result

This gives more information but still does not provide one stable identifier
for the complete experiment.

### Use an explicit run identifier

This is slightly more code, but it makes API queries, dashboard filtering, and
baseline-versus-federated comparisons unambiguous.

## Consequences

- New experiment results can be shown as one coherent run.
- The dashboard can later provide a run-history selector.
- Existing historical rows remain visible but are not comparable as one run.
- Training scripts must explicitly create, complete, or fail a run.
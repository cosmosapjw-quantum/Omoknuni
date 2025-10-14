# Archived Documents

**Archive Date**: 2025-10-14
**Reason**: Superseded by v2.0 specification documents

---

## Archived Files

These documents have been superseded by newer, consolidated versions:

| Archived File | Superseded By | Reason |
|---------------|---------------|--------|
| **SPECIFICATION.md** (1,210 lines) | [spec.md](../spec.md) v2.0 (503 lines) | Consolidated functional spec with review.txt integration |
| **TASKS.md** (1,751 lines) | [tasks.md](../tasks.md) v1.0 (2,747 lines) | Enhanced with CLARIFICATIONS.md resolutions and detailed acceptance tests |
| **TECHNICAL_PLAN.md** (1,034 lines) | [plan.md](../plan.md) v1.0 (2,398 lines) | Expanded with architecture diagrams, code snippets, and precise insertion points |
| **PR_CHECKLIST.md** (509 lines) | [ACCEPTANCE_CHECKLIST.md](../ACCEPTANCE_CHECKLIST.md) (860 lines) | Comprehensive 150+ check validation suite with critical/target/stretch criteria |
| **REVIEW-RESPONSE.md** (183 lines) | [TRACEABILITY_MATRIX.md](../TRACEABILITY_MATRIX.md) (524 lines) | Full cross-check analysis replacing one-time response |

---

## What Changed

### SPECIFICATION.md → spec.md v2.0

**Improvements**:
- Reduced from 1,210 to 503 lines (58% reduction) via consolidation
- Added Multi-Actor Self-Play requirements (G7, Section 4.2)
- Integrated review.txt findings (7.5ms feature extraction, 2-3× cloning, 60% idle)
- Added measurable KPIs with acceptance criteria
- Removed duplicate content (moved to plan.md and data-model.md)

**Migration Notes**:
- All functional requirements preserved
- Goals (G1-G8) enhanced with quantitative targets
- User stories (US1-US3) added for clarity

---

### TASKS.md → tasks.md v1.0

**Improvements**:
- Expanded from 1,751 to 2,747 lines (57% growth) with detail
- Added dependency graph visualization
- Included precise code snippets for all tasks
- Enhanced acceptance tests with unit + performance validation
- Added "Done means" criteria with telemetry fields
- Integrated CLARIFICATIONS.md resolutions (OpenMP, state pooling, CV, allocator, cache, multi-actor)

**Migration Notes**:
- All 30 tasks preserved
- Phase structure unchanged (0: Foundation, 1: CPU, 2: Validation, 3: NN-Cache, 4: Multi-Actor, 5: Docs)
- Added T001 (Benchmark Harness) as foundation task

---

### TECHNICAL_PLAN.md → plan.md v1.0

**Improvements**:
- Expanded from 1,034 to 2,398 lines (132% growth)
- Added 2 Mermaid architecture diagrams (Single-MCTS, Multi-Actor)
- Included implementation code snippets for all optimizations
- Added precise file:line insertion points
- Enhanced with risk mitigation and rollback procedures

**Migration Notes**:
- All sections preserved: A) Architecture, B) CPU Pipeline, C) NN-Cache, D) Multi-Actor, E) Telemetry, F) Risk
- Added B1.1 diagnosis showing OpenMP already in code
- Added C) NN-Eval Cache complete design (Zobrist, sharding, eviction)

---

### PR_CHECKLIST.md → ACCEPTANCE_CHECKLIST.md

**Improvements**:
- Expanded from 509 to 860 lines (69% growth)
- Structured into 3 categories: Performance (60 checks), Correctness (45 checks), Operational (45 checks)
- Added critical path (11 MUST PASS checks)
- Included specific commands for each validation
- Added rollback triggers and procedures

**Migration Notes**:
- All original checklist items preserved
- Enhanced with evidence requirements (CSV, plots, logs)
- Added telemetry field validation

---

### REVIEW-RESPONSE.md → TRACEABILITY_MATRIX.md

**Improvements**:
- One-time response replaced by comprehensive traceability analysis
- 100% coverage of review.txt bottlenecks → spec → plan → tasks
- Identified 3 gaps (all non-blocking, documented)
- 0 contradictions found across all documents
- Provides ongoing validation (not just initial response)

**Migration Notes**:
- Original review points all addressed in matrix
- Added gap analysis with recommendations
- Included completeness check and final verdict

---

## Reference

For the current, authoritative documentation, see:
- **[../README.md](../README.md)** - Spec 004 overview
- **[../spec.md](../spec.md)** - Functional requirements
- **[../plan.md](../plan.md)** - Technical implementation
- **[../tasks.md](../tasks.md)** - Task breakdown
- **[../ACCEPTANCE_CHECKLIST.md](../ACCEPTANCE_CHECKLIST.md)** - Validation checklist

---

**Note**: These archived files are kept for historical reference only. Do not use for implementation.

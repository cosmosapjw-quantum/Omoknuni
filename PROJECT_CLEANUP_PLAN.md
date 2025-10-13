# Project-Wide Documentation Cleanup and Update Plan

**Date**: 2025-10-13
**Purpose**: Align all project documentation with authoritative specs/004 documents
**Authority**: CONSTITUTION.md v1.1, SPECIFICATION.md v1.1, TECHNICAL_PLAN.md v2.0, TASKS.md v2.1

---

## Executive Summary

This plan comprehensively cleans up legacy files and updates all project documentation to reflect the **revised 8k sims/sec target** (hardware-grounded, validated 2025-10-13) and current Spec 004 validation status.

**Key Changes**:
- Remove all references to 25k/30k outdated targets
- Update to 8k sims/sec target (realistic, hardware-grounded)
- Delete 9 legacy/temporary files from specs/004
- Update API contracts for consistency
- Synchronize all project documentation

---

## Phase 1: Delete Legacy Files from specs/004 (9 files)

### Files to DELETE:

1. **plan-LEGACY-v1.0.md**
   - Reason: Superseded by TECHNICAL_PLAN.md v2.0
   - Status: Contains outdated 25k target

2. **tasks-phases-1-3-ARCHIVE.md**
   - Reason: Superseded by TASKS.md v2.1
   - Status: Historical archive only

3. **gap-analysis-HISTORICAL.md**
   - Reason: Historical analysis, no longer relevant
   - Status: Marked as HISTORICAL in filename

4. **ANALYSIS_INCONSISTENCIES.md**
   - Reason: Temporary analysis from spec kit work
   - Status: Analysis complete, findings applied

5. **ANALYSIS_FIXES_APPLIED.md**
   - Reason: Temporary tracking of fixes applied
   - Status: All fixes complete

6. **UPDATE_PLAN.md**
   - Reason: Temporary planning document
   - Status: Plan execution complete

7. **ULTRATHINK_COMPREHENSIVE_UPDATE_SUMMARY.md**
   - Reason: Temporary summary of comprehensive update
   - Status: Update complete, findings in SPEC_UPDATE_COMPLETION_REPORT

8. **SPEC_UPDATE_COMPLETION_REPORT.md**
   - Reason: Temporary completion report
   - Status: Report complete, can be deleted or archived

9. **SPECKIT_ANALYSIS_2025-10-13.md**
   - Reason: Temporary spec kit analysis
   - Status: Analysis complete, single fix applied

**Action**: `rm` all 9 files

---

## Phase 2: Update Project Root Documentation

### 2.1 README.md (CRITICAL - Multiple outdated references)

**Current Issues**:
- Line 19: "Target: 25,000+ sims/sec (11.6× current performance)"
- Line 22: "Expected Path: Current 2,147 → validate optimizations → expected 18-36k → +tuning: ≥25k sims/sec"
- Line 54: "Theoretical Maximum: ~4,167 sims/sec even with instant GPU (7.2× below target)"
- Line 62-66: Outdated "Conclusion" and "Next Steps" sections
- Line 78: "Goal: Achieve 25,000+ simulations/second (6.8× improvement)"

**Updates Required**:
- Replace all "25,000", "25k", "30,000", "30k" with "8,000 sims/sec (realistic, hardware-grounded)"
- Update validation status: T-VALID-1 PASS, T-VALID-2 FAIL
- Update critical path: OpenMP fix required
- Update Phase 2 status: COMPLETE and VALIDATED (not "UNVALIDATED")
- Update expected performance path
- Update conclusion to match CONSTITUTION.md Section 1.3

### 2.2 CLAUDE.md (Partially updated, needs verification)

**Current Status**: Already has validation results (2025-10-13), but may have stale references

**Updates Required**:
- Line 16: "targets 30-40k simulations/second" → "targets 8,000 sims/sec (realistic, hardware-grounded)"
- Line 39: "ready for 30k+ with GPU batching" → "targeting 8k sims/sec with full optimization"
- Verify Performance Targets table matches CONSTITUTION.md
- Verify all spec status matches TASKS.md v2.1

### 2.3 docs/mcts_cpp_runner.md

**Current Issues**:
- Line 7: "30-40k simulations/second (vs 246 sims/sec Python baseline)"

**Updates Required**:
- Replace "30-40k" with "8,000 sims/sec (realistic target)"
- Add note about hardware limits (RTX 3060 Ti @ FP16)
- Update performance characteristics section if needed

### 2.4 Other Root Files (mcts_guide.md, PROFILING_DESIGN.md, etc.)

**Files to Check**:
- mcts_guide.md
- PROFILING_DESIGN.md
- HOWTO-RUN-TESTS.md
- CHANGELOG.md
- AGENTS.md

**Action**: Search for "25k", "30k", "25000", "30000" and replace with 8k target

---

## Phase 3: Update docs/ Directory

### Files Needing Updates:

1. **docs/api.md** - API documentation
2. **docs/operations.md** - Operations guide
3. **docs/profiling_framework.md** - Profiling framework
4. **docs/spec_kit_analysis_summary.md** - Spec kit analysis summary
5. **docs/training_guide.md** - Training guide
6. **docs/alpha_beta_testing_guide.md** - Testing guide

**Updates Required**:
- Replace all 25k/30k references with 8k target
- Update validation status if mentioned
- Update critical path if mentioned
- Ensure consistency with CONSTITUTION.md

---

## Phase 4: Review and Update API Contracts

### Contracts Already Updated (Verified 2025-10-13):
- ✅ dlpack_api.yaml
- ✅ dlpack-api.md
- ✅ inference_api.yaml

### Contracts Needing Review:

1. **arena-api.md**
   - Check: Performance targets, thread count assumptions
   - Update: If referencing 25k/30k targets

2. **python-bridge-api.md**
   - Check: Throughput targets, performance benchmarks
   - Update: If referencing outdated targets

3. **queue_api.yaml**
   - Check: Queue size rationale (currently 4096 for 30k target?)
   - Update: If sizing based on outdated targets

4. **thread_api.yaml**
   - Check: Thread count recommendations
   - Update: If based on 25k/30k targets

5. **virtual_loss_api.yaml**
   - Check: Virtual loss magnitude assumptions
   - Update: If calibrated for outdated targets

---

## Phase 5: Verify specs/004 Supporting Docs

### Documents to Verify (Should be current):

1. **spec.md** - Overview (updated 2025-10-13) ✅
2. **research.md** - Technical analysis (updated 2025-10-13) ✅
3. **data-model.md** - Data structures (updated 2025-10-13) ✅
4. **quickstart.md** - Build guide (updated 2025-10-13) ✅
5. **REVIEW-RESPONSE.md** - Review response (updated 2025-10-13) ✅
6. **README.md** - Spec 004 overview

**Action**: Quick verification pass to ensure consistency

---

## Phase 6: Final Verification

### Consistency Checks:

1. **Performance Targets**: All files reference 8k sims/sec (6k minimum, 10k stretch)
2. **Validation Status**: T-VALID-1 PASS, T-VALID-2 FAIL consistently reported
3. **Critical Blocker**: OpenMP fix at dlpack_bridge.cpp:431-434 mentioned
4. **Hardware Specs**: AMD Ryzen 9 5900X + RTX 3060 Ti (8GB)
5. **Constitutional Constraints**: NO libtorch/TensorRT/ONNX consistently enforced

### Final Actions:

1. Delete PROJECT_CLEANUP_PLAN.md (this file) after execution
2. Create CLEANUP_COMPLETION_REPORT.md documenting all changes
3. Git status check for modified files

---

## Success Criteria

- ✅ All 9 legacy files deleted from specs/004
- ✅ All references to 25k/30k targets updated to 8k
- ✅ All API contracts reviewed and consistent
- ✅ All project documentation synchronized with CONSTITUTION.md v1.1
- ✅ Zero contradictions between any documentation files
- ✅ Clear communication of validation status and critical blocker

---

## Timeline

- Phase 1 (Delete): 5 minutes
- Phase 2 (Root docs): 20 minutes
- Phase 3 (docs/): 15 minutes
- Phase 4 (API contracts): 15 minutes
- Phase 5 (Verify specs/004): 5 minutes
- Phase 6 (Final verification): 10 minutes

**Total**: ~70 minutes

# Requirements and Priority Guide

This guide turns the broader SDLC handoff into a concise product checklist. Use it to align delivery planning, implementation sequencing, and stakeholder conversations around what is required for the MVP versus what can follow later.

## Priority definitions

| Priority | Meaning | Delivery expectation |
|---|---|---|
| Required | Needed for the Production MVP to satisfy the core prediction, governance, and operational contract. | Must be implemented or explicitly waived before MVP promotion. |
| High | Strongly recommended for pilot readiness or safe production operation. | Should be planned immediately after the required MVP scope if not included in the first release. |
| Nice to have | Valuable enhancement that improves usability, automation, or analytical depth but is not needed for the first decision-support release. | Can be deferred until controlled pilot, production scale, or advanced capability phases. |

## Required MVP features

| Area | Requirement | Source requirement |
|---|---|---|
| Loan scoring | Score all eligible open loans. | FR-001 |
| Prediction horizons | Produce recall probabilities for 1, 3, 5, and 10 business-day horizons. | FR-002 |
| Risk classification | Assign Low, Moderate, High, and Critical risk bands. | FR-003 |
| Explainability | Provide at least three reason codes per prediction. | FR-004 |
| API access | Expose predictions through a secure API. | FR-005 |
| Prediction storage | Store predictions, features, model version, and outcomes for audit, monitoring, and replay. | FR-006 |
| Backtesting | Support replay and historical backtesting. | FR-007 |
| Configurable thresholds | Support configurable alert thresholds. | FR-009 |
| Duplicate controls | Suppress duplicate alerts. | FR-010 |
| Daily operations | Complete daily scoring before the agreed market cut-off. | NFR-001 |
| Reproducibility | Make predictions reproducible from versioned data and code. | NFR-003 |
| Access control | Protect sensitive data with appropriate access controls. | NFR-004 |
| Data quality and lineage | Provide data lineage and data-quality checks for the scoring pipeline. | NFR-005 |
| Safe operation | Support rollback and safe degradation when data is delayed. | NFR-006, NFR-007 |
| Monitoring | Detect model, feature, and pipeline failures. | NFR-008 |

## High-priority follow-up features

These items are not all required to prove the MVP contract, but they materially improve pilot readiness, operational safety, and model governance.

| Area | Feature | Why it matters |
|---|---|---|
| User action audit | Capture an auditable record of user actions. | Required by FR-008 and important once users begin acting on predictions. |
| Dashboard | Launch an operational dashboard for risk review and stakeholder adoption. | Enables shadow mode and controlled pilot workflows. |
| Model registry | Register promoted models and associated metadata. | Supports controlled deployment, rollback, and auditability. |
| Artifact integrity | Add deterministic model content, checksums, and richer metadata. | Strengthens reproducibility and promotion controls. |
| API validation policy | Clarify validation versus fallback behavior for invalid feature rows. | Prevents silent scoring surprises in production. |
| API integration tests | Cover scorer-backed and fallback-backed API paths. | Locks down expected runtime behavior. |
| Batch scoring | Provide a command-line batch scoring path from persisted artifacts. | Supports offline scoring, operations, and recovery workflows. |
| Structured observability | Add structured logging and metrics. | Helps operators distinguish artifact scoring from fallback scoring and diagnose failures. |
| Compatibility tests | Add golden artifact compatibility tests. | Prevents accidental breaking changes to saved model artifacts. |
| Promotion and reload policy | Decide and document model promotion and scorer reload behavior. | Clarifies operational ownership for model rotation. |

## Nice-to-have and later-phase capabilities

| Capability | Suggested phase | Notes |
|---|---|---|
| Controlled user pilot workflow | Controlled Pilot | Release to limited users, capture feedback and actions, tune thresholds, and address false alerts. |
| Expanded users and markets | Production Scale | Broaden the footprint after monitoring, support, and governance are formalized. |
| Integrated alerts | Production Scale | Push predictions into operational alert channels once precision, capacity, and escalation rules are accepted. |
| Optimizer inputs | Production Scale | Feed recall risk into allocation, inventory, or pricing optimization after model-risk review. |
| Survival modeling | Advanced Capabilities | Useful for modeling time-to-recall beyond fixed classification horizons. |
| Recall quantity prediction | Advanced Capabilities | Estimates expected recalled quantity, not just recall probability. |
| Event-driven scoring | Advanced Capabilities | Scores loans intra-day or in response to events when daily batch cadence is insufficient. |
| Graph features | Advanced Capabilities | Uses relationships among lenders, borrowers, securities, and market behaviors. |
| Automated decision support | Advanced Capabilities | Moves beyond recommendations toward direct workflow or optimizer influence; should follow stronger governance. |
| Numeric contribution details | Advanced Capabilities | Adds analyst/audit drill-through details alongside business-friendly reason codes. |

## Suggested implementation sequence

1. Confirm MVP scope, label definition, eligible-loan population, prediction horizons, and intervention workflow.
2. Complete the required MVP API, scoring, storage, backtesting, data-quality, monitoring, rollback, and access-control capabilities.
3. Add the high-priority pilot-readiness items, especially API validation policy, artifact integrity, integration tests, batch scoring, observability, and model promotion policy.
4. Run shadow mode with dashboard review and no automated downstream action.
5. Move into a controlled pilot with limited users, captured actions, measured business value, and tuned thresholds.
6. Defer advanced modeling, optimizer integration, event-driven scoring, and automated decision support until production behavior and governance are proven.

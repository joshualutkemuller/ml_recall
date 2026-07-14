# Loan Recall Prediction Platform
## Quantitative Research, Product, Engineering, and Model Risk Handoff Specification

**Document purpose:** Define an execution-ready plan for building, validating, deploying, and operating a machine learning system that predicts the probability and timing of securities lending loan recalls.

**Primary users:** Quantitative researchers, data scientists, platform engineers, securities lending product managers, trading desks, operations teams, model risk, compliance, data governance, and production support.

**System objective:** For every active securities lending loan, estimate the probability that the lender will issue a recall within specified horizons and convert that probability into operationally useful alerts, inventory actions, and optimizer inputs.

---

# Implementation Handoff Update — 2026-07-14

The current MVP implementation has completed the requested Stage 3 foundation items:

- Added point-in-time loan feature engineering that computes loan age, utilization ratio, lender recent recall activity, lender recall rate, and borrow-fee percentile while excluding recall and market rows after each scoring row's `as_of_timestamp` to prevent look-ahead leakage.
- Added reusable scoring-row data quality checks with structured PASS/FAIL status, errors, warnings, duplicate loan/as-of detection, required-field validation, and non-blocking negative inventory warnings.
- Added unit coverage for point-in-time feature behavior, including future event exclusion and engineered feature values.
- Added unit coverage for data quality validation failures and non-blocking warnings.
- Updated the README MVP scope to include point-in-time feature engineering and scoring data quality checks.
- Tightened model artifact reproducibility semantics by removing generated wall-clock artifact fields, adding explicit dataset/label/window/code/seed reproducibility metadata, and validating a canonical SHA-256 artifact digest.

---

# 1. Executive Summary

Loan recalls create operational risk, potential buy-ins, settlement failures, lost revenue, and client disruption. Most firms manage recall risk reactively using rules, trader experience, and manual monitoring. The proposed system will produce calibrated recall probabilities at the loan level and make those predictions available to traders, operations teams, inventory managers, and collateral or allocation optimizers.

The initial production model should not attempt to automate every downstream action. The first release should operate as a decision-support system that:

1. Scores all eligible open loans daily and intraday where data permits.
2. Predicts recall risk over 1-day, 3-day, 5-day, and 10-day horizons.
3. Provides reason codes and uncertainty indicators.
4. Surfaces the highest-risk loans through APIs, dashboards, and workflow alerts.
5. Supports controlled use by inventory allocation and optimization engines.
6. Captures user actions and outcomes to create a continuous learning loop.

The key design principle is that the prediction itself is not the product. The product is an intervention workflow that reduces avoidable fails, buy-ins, emergency sourcing, and revenue leakage without overwhelming users with false alerts.

---

# 2. Problem Definition

## 2.1 Business Problem

A lender may recall a securities loan for reasons including:

- Portfolio sale activity
- Proxy voting or corporate governance events
- Fund redemptions
- Corporate actions
- Internal liquidity needs
- Benchmark or index changes
- Portfolio rebalancing
- Changes in lending program policy
- Tax or dividend considerations
- Counterparty or concentration limits

Recalls may require a borrower or agent lender to:

- Source replacement inventory
- Close or rebook the loan
- Reallocate inventory across clients
- Adjust collateral
- Manage settlement risk
- Escalate operational exceptions
- Update pricing or availability

The absence of an early-warning signal forces users into reactive behavior.

## 2.2 ML Problem Statement

For each active loan \(i\) at time \(t\), estimate:

\[
P(Y_{i,t,h}=1 \mid X_{i,t})
\]

where:

- \(Y_{i,t,h}=1\) if the loan is recalled within horizon \(h\)
- \(h \in \{1,3,5,10\}\) business days
- \(X_{i,t}\) is the feature set available at prediction time

Secondary objectives:

- Estimate expected time to recall
- Rank loans by near-term recall risk
- Predict whether a recall is partial or full
- Estimate probable recalled quantity
- Predict recall operational severity

## 2.3 Recommended Modeling Formulation

Use a layered approach:

### Phase 1
Binary classification for each time horizon.

### Phase 2
Discrete-time survival model or hazard model.

### Phase 3
Multi-task model predicting:

- Recall probability
- Time to recall
- Recall quantity
- Operational severity
- Confidence or uncertainty

The first production model should favor interpretability and calibration over architectural novelty.

---

# 3. Product Scope

## 3.1 In Scope

- Open equity and fixed-income securities loans with sufficient historical data
- Loan-level recall prediction
- Multiple prediction horizons
- Daily batch scoring
- Optional intraday refresh
- Feature engineering pipeline
- Explainability and reason codes
- Risk dashboard
- Prediction API
- Alerting workflow
- Model monitoring
- Outcome capture
- Feedback from users
- Integration with inventory and optimization systems
- Model governance artifacts
- Audit logging

## 3.2 Out of Scope for Initial Release

- Fully autonomous loan termination
- Fully autonomous inventory substitution
- Client-facing predictions without review
- Use of unapproved alternative data
- Cross-border legal interpretation
- Dynamic pricing changes based solely on model output
- Replacement of existing operational controls
- Reinforcement learning in production
- End-to-end optimization driven entirely by predicted recalls

## 3.3 Success Criteria

The model is successful only if it improves business outcomes.

Primary success measures:

- Reduction in avoidable settlement failures
- Reduction in emergency sourcing events
- Reduction in recall-related buy-ins
- Reduction in manual review time
- Earlier median warning time
- Improvement in inventory allocation
- Reduction in high-severity recall incidents
- Positive user adoption and action rate

Supporting ML measures:

- Precision at top \(K\)
- Recall at top \(K\)
- PR-AUC
- Brier score
- Calibration error
- Lift over rules baseline
- Stability across clients, markets, and regimes

---

# 4. Stakeholders and Roles

| Role | Primary Responsibility |
|---|---|
| Executive Sponsor | Funding, prioritization, escalation |
| Product Owner | Business requirements, roadmap, adoption |
| Quant Lead | Model design, research, validation |
| Data Science Lead | Feature engineering, training, experimentation |
| Data Engineering | Pipelines, data quality, lineage |
| Platform Engineering | APIs, orchestration, production infrastructure |
| Securities Lending Desk | Domain expertise, workflow validation |
| Operations | Recall handling, controls, escalation design |
| Inventory Management | Replacement sourcing and allocation use cases |
| Model Risk | Independent validation and approval |
| Compliance and Legal | Use restrictions, fairness, conduct review |
| Data Governance | Data ownership, retention, quality |
| Information Security | Access controls, vulnerability review |
| Production Support | Incident response, SLAs, runbooks |
| Internal Audit | Traceability and control assurance |

## 4.1 RACI Summary

| Workstream | Product | Quant | Data Eng | Platform Eng | Desk/Ops | Model Risk |
|---|---:|---:|---:|---:|---:|---:|
| Requirements | A | R | C | C | R | C |
| Label design | C | A/R | C | C | R | C |
| Data pipeline | C | C | A/R | C | C | C |
| Model development | C | A/R | C | C | C | C |
| API and UI | A | C | C | R | C | C |
| Validation | C | R | C | C | C | A/R |
| Deployment | C | C | C | A/R | C | C |
| Monitoring | A | R | R | R | C | C |
| Production approval | A | R | C | R | C | A |

---

# 5. Six-Stage Software Development Lifecycle

The project will follow six formal stages. Each stage has required inputs, activities, deliverables, acceptance criteria, and handoff gates.

---

# Stage 1 — Discovery and Requirements

## 5.1 Objectives

- Define the business decision the model will support.
- Establish the population, prediction horizons, labels, and users.
- Quantify economic value.
- Document constraints, risks, and governance requirements.
- Identify available data and data gaps.

## 5.2 Key Questions

- Who issues recalls and through what channels?
- What constitutes a recall event?
- How are partial recalls represented?
- What timestamp should define the event?
- How much warning time is operationally useful?
- Which loans should be excluded?
- What action should a user take when risk is high?
- What is the cost of a false positive?
- What is the cost of a missed recall?
- How should risk thresholds vary by security, client, or desk?

## 5.3 Requirements Specification

### Functional Requirements

**FR-001:** The system shall score all eligible open loans.

**FR-002:** The system shall produce recall probabilities for 1, 3, 5, and 10 business-day horizons.

**FR-003:** The system shall produce a risk band: Low, Moderate, High, Critical.

**FR-004:** The system shall provide at least three reason codes per prediction.

**FR-005:** The system shall expose predictions through a secure API.

**FR-006:** The system shall store predictions, features, model version, and outcome.

**FR-007:** The system shall support replay and historical backtesting.

**FR-008:** The system shall provide an auditable record of user actions.

**FR-009:** The system shall support configurable alert thresholds.

**FR-010:** The system shall suppress duplicate alerts.

### Nonfunctional Requirements

**NFR-001:** Daily scoring shall complete before the agreed market cut-off.

**NFR-002:** API response latency shall meet platform requirements.

**NFR-003:** Predictions shall be reproducible from versioned data and code.

**NFR-004:** All sensitive data shall be access-controlled.

**NFR-005:** The pipeline shall provide data lineage and quality checks.

**NFR-006:** The system shall support rollback to the prior model.

**NFR-007:** The service shall degrade safely if data is delayed.

**NFR-008:** Monitoring shall detect model, feature, and pipeline failures.

## 5.4 Economic Value Framework

Estimate expected value:

\[
EV = \sum_i \left[
P_i(\text{recall}) \times B_i(\text{successful intervention})
- C_i(\text{false alert})
- C_i(\text{intervention})
\right]
\]

Potential benefit components:

- Avoided fail cost
- Avoided buy-in cost
- Avoided emergency borrow premium
- Preserved client revenue
- Reduced operations effort
- Reduced market impact
- Reduced capital or liquidity usage

## 5.5 Discovery Deliverables

- Product requirements document
- Business process map
- Recall event taxonomy
- Data inventory
- Label definition document
- Economic value model
- Stakeholder map
- Initial risk register
- Governance assessment
- Feasibility recommendation

## 5.6 Stage 1 Exit Criteria

- Business owner approves use case.
- Label is unambiguously defined.
- Required data sources are identified.
- Economic value is material.
- Users agree on proposed interventions.
- Model risk classification is established.
- MVP scope is approved.

---

# Stage 2 — System and Model Design

## 6.1 Objectives

- Design the target architecture.
- Define data contracts.
- Select baseline and candidate model families.
- Design features, training splits, validation, explainability, and downstream integrations.
- Define controls before coding begins.

## 6.2 Target Architecture

```text
Source Systems
    |
    v
Raw Data Layer
    |
    v
Validated Historical Tables
    |
    v
Feature Engineering Pipeline
    |
    +--> Offline Feature Store
    |
    +--> Online/Serving Feature Store
    |
    v
Training Pipeline
    |
    v
Model Registry
    |
    v
Batch Scoring / Real-Time Scoring
    |
    +--> Prediction Store
    +--> API
    +--> Dashboard
    +--> Alerts
    +--> Optimizer Inputs
    |
    v
Outcome and Feedback Capture
    |
    v
Monitoring and Retraining
```

## 6.3 Data Domains

### Loan Data

- Loan identifier
- Security identifier
- Lender
- Borrower
- Agent
- Open date
- Quantity
- Market value
- Fee or rebate
- Currency
- Loan type
- Term or open status
- Current status
- Collateral type
- Recall history
- Return history

### Security Data

- Asset class
- Sector
- Country
- Exchange
- Market capitalization
- Liquidity
- Volatility
- Corporate actions
- Dividend dates
- Proxy dates
- Index membership
- Short interest
- Utilization
- Days to cover
- Borrow fee
- Specialness indicators

### Lender and Fund Data

- Fund type
- Historical recall behavior
- Redemption patterns
- Turnover
- Voting policy
- Concentration
- Assets under management
- Lending utilization
- Seasonal behavior

### Market Data

- Price returns
- Volatility
- Volume
- spreads
- ETF flows
- Index rebalance calendar
- Market stress indicators
- Funding rates
- Settlement fail indicators

### Operational Data

- Prior fails
- Manual exceptions
- Settlement location
- Cut-off times
- Message timestamps
- Processing delays
- Desk actions
- Inventory availability

## 6.4 Label Specification

### Primary Label

For loan \(i\), prediction timestamp \(t\), and horizon \(h\):

```text
label_recall_h = 1
if a valid lender-initiated recall occurs after t and on or before t+h
otherwise 0
```

### Exclusions

Exclude or separately model:

- Borrower-initiated returns
- System-generated closures
- Maturity events
- Corporate action mandatory closures
- Data corrections
- Duplicate recalls
- Cancelled recalls
- Loans already under recall
- Loans with incomplete event histories

### Partial Recalls

Create additional fields:

- `partial_recall_flag`
- `recall_quantity`
- `recall_percentage`
- `full_recall_flag`

## 6.5 Leakage Controls

Prohibited features include:

- Recall status populated after the decision timestamp
- Messages received after prediction time
- Future corporate action processing fields
- Manual status flags entered in response to the recall
- Loan close reason if populated after event
- Any aggregate using future observations

Every feature must include:

- Source timestamp
- Effective timestamp
- Ingestion timestamp
- Availability timestamp
- Point-in-time correctness rule

## 6.6 Candidate Models

### Baselines

- Rules-based score
- Logistic regression
- Historical lender recall rate
- Security-level recall frequency
- Recency-weighted empirical probability

### Candidate ML Models

- Regularized logistic regression
- Random forest
- XGBoost
- LightGBM
- CatBoost
- Discrete-time hazard model
- Cox proportional hazards model
- Random survival forest

### Advanced Candidates

- Temporal Fusion Transformer
- Sequence models
- Graph-based lender-security models
- Multi-task neural networks

Initial production selection should likely be logistic regression or gradient boosting with calibration.

## 6.7 Evaluation Design

### Time-Based Splits

Use:

- Training period
- Validation period
- Holdout test period
- Recent regime stress period

Do not randomly split observations across time.

### Entity Leakage Controls

Evaluate whether the same:

- Lender
- Fund
- Security
- Borrower
- Loan lineage

appears across train and test.

Create additional cold-start tests for unseen lenders or securities where material.

### Core Metrics

- PR-AUC
- ROC-AUC
- Precision at top 1%, 5%, and 10%
- Recall at top 1%, 5%, and 10%
- Lift over baseline
- Brier score
- Expected calibration error
- Log loss
- Alert volume
- Average lead time
- Business value captured

### Threshold Selection

Thresholds should optimize expected economic value, not F1 score alone.

## 6.8 Design Deliverables

- Solution architecture
- Data model
- Data contracts
- Feature catalog
- Label specification
- Modeling plan
- Experiment plan
- Validation plan
- Security design
- API specification
- UI wireframes
- Test strategy
- Monitoring design
- Deployment design

## 6.9 Stage 2 Exit Criteria

- Architecture is approved.
- Data contracts are signed off.
- Leakage review is complete.
- Baseline model is defined.
- Evaluation framework is approved.
- Security and access model is approved.
- Test and monitoring plans are complete.

---

# Stage 3 — Development and Implementation

## 7.1 Objectives

- Build reproducible data, feature, training, validation, and scoring pipelines.
- Implement interfaces, storage, and workflow components.
- Establish coding, testing, and documentation standards.

## 7.2 Repository Structure

```text
loan-recall-prediction/
├── README.md
├── pyproject.toml
├── environment.yml
├── Makefile
├── configs/
│   ├── base.yaml
│   ├── dev.yaml
│   ├── test.yaml
│   └── prod.yaml
├── docs/
│   ├── product-requirements.md
│   ├── architecture.md
│   ├── model-card.md
│   ├── data-dictionary.md
│   ├── runbook.md
│   └── change-log.md
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── labels/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   ├── scoring/
│   ├── api/
│   ├── monitoring/
│   └── common/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   ├── data_quality/
│   └── model/
├── notebooks/
│   ├── exploration/
│   └── archived/
├── pipelines/
├── infrastructure/
├── sql/
├── schemas/
└── artifacts/
```

## 7.3 Engineering Standards

- Python type hints required
- Linting and formatting enforced
- Unit test coverage targets
- No business logic in notebooks
- Configuration separated from code
- Secrets stored in approved vault
- Feature definitions versioned
- Data schemas enforced
- Deterministic random seeds
- Immutable model artifacts
- Structured logging
- Dependency pinning
- Pre-commit hooks
- Mandatory peer review
- CI required for merge

## 7.4 Feature Pipeline Specification

Each feature shall have:

| Field | Description |
|---|---|
| Feature name | Stable identifier |
| Business definition | Meaning |
| Source system | Origin |
| Source field | Raw field |
| Transformation | Calculation |
| Lookback window | Historical window |
| Availability lag | Delay |
| Null policy | Imputation |
| Valid range | Data quality rule |
| Owner | Data steward |
| Version | Definition version |

## 7.5 Experiment Tracking

Track:

- Dataset version
- Feature set version
- Label version
- Hyperparameters
- Code commit
- Training window
- Evaluation window
- Metrics
- Calibration method
- Model artifact
- Approver
- Notes and decision

## 7.6 API Contract

### Endpoint

`POST /v1/loan-recall/predict`

### Request

```json
{
  "loan_ids": ["LN123", "LN456"],
  "as_of_timestamp": "2026-07-10T13:00:00Z",
  "horizons": [1, 3, 5, 10]
}
```

### Response

```json
{
  "model_version": "recall_model_1.0.0",
  "predictions": [
    {
      "loan_id": "LN123",
      "as_of_timestamp": "2026-07-10T13:00:00Z",
      "probabilities": {
        "1d": 0.12,
        "3d": 0.27,
        "5d": 0.41,
        "10d": 0.63
      },
      "risk_band": "HIGH",
      "reason_codes": [
        "LENDER_RECENT_RECALL_ACTIVITY",
        "UPCOMING_PROXY_EVENT",
        "HIGH_FUND_TURNOVER"
      ],
      "data_quality_status": "PASS"
    }
  ]
}
```

## 7.7 User Interface Requirements

Dashboard views:

- Highest-risk open loans
- Recall risk by lender
- Recall risk by security
- Risk by market and asset class
- Upcoming corporate action concentration
- Recent alerts
- User action status
- Prediction explanations
- Model health
- Data freshness

## 7.8 PowerBI Analytics Output Handoff

The analytics layer must treat PowerBI as a first-class downstream consumer rather than an afterthought. Before model outputs are promoted beyond development, the product, data engineering, BI, quant, model-risk, and operations owners must complete the following decisions and implementation steps.

### Required decisions before building PowerBI ingestion

| Decision | Owner | Required output |
|---|---|---|
| Business grain | Product, BI, Quant | Confirm whether reporting is loan-as-of, loan-horizon-as-of, alert, action, or outcome grain. The recommended default is one row per `loan_id`, `as_of_date`, and `horizon_days` in the PowerBI fact table. |
| Refresh cadence | Product, Data Eng | Select daily-only, intraday snapshot, or streaming/near-real-time refresh and document the freshness SLA displayed in reports. |
| Semantic model mode | BI, Data Eng | Choose Import, DirectQuery, composite model, or Fabric semantic model based on volume, latency, and entitlement requirements. |
| Risk thresholds | Product, Desk, Model Risk | Approve the Low, Moderate, High, and Critical thresholds for each horizon and whether thresholds vary by desk, lender, asset class, or region. |
| Security model | Compliance, BI, Data Governance | Define row-level security by desk, region, client, lender, borrower, portfolio, and model-risk audience. |
| Explainability display | Quant, Model Risk, Product | Approve which reason codes, feature contributions, uncertainty flags, and model limitations may be shown to each user group. |
| Action taxonomy | Operations, Product | Define valid user actions, escalation statuses, suppression reasons, and closed-loop outcome codes. |
| Certified measures | Finance, Product, BI | Approve business-value formulas for avoided fail cost, emergency sourcing cost, buy-in exposure, and intervention cost. |
| Historical retention | Data Governance | Define retention, partitioning, and archival policy for prediction snapshots, features, alerts, actions, and outcomes. |
| Reconciliation control | Data Eng, BI | Define row-count, freshness, schema, and aggregate reconciliation checks between the prediction store and PowerBI model. |

### Model-output contract for PowerBI

Publish a curated analytics table or view rather than connecting reports directly to raw API responses. The minimum PowerBI-ready prediction fact should include:

| Field | Purpose |
|---|---|
| `prediction_id` | Stable unique key for a scored loan-horizon snapshot. |
| `loan_id` | Join key to loan and action dimensions. |
| `security_id` | Join key to security attributes. |
| `lender_id` | Join key to lender attributes and row-level security rules. |
| `borrower_id` | Join key to borrower, desk, or client coverage. |
| `as_of_timestamp_utc` | Exact model scoring timestamp. |
| `as_of_date` | Date key for PowerBI relationships and incremental refresh. |
| `horizon_days` | Prediction horizon, such as 1, 3, 5, or 10 business days. |
| `recall_probability` | Calibrated probability used for ranking, bands, and expected-value measures. |
| `risk_band` | Approved business band: Low, Moderate, High, or Critical. |
| `risk_score_rank` | Optional same-day percentile or rank for top-K reporting. |
| `reason_code_1` through `reason_code_5` | Ordered reason codes for report explanations. |
| `top_feature_1` through `top_feature_5` | Optional user-approved feature names for explainability. |
| `data_quality_status` | PASS, WARN, or FAIL. FAIL rows should be hidden from operational pages by default. |
| `model_version` | Version used to score the row. |
| `feature_set_version` | Feature definition version. |
| `calibration_version` | Calibration artifact version. |
| `alert_generated_flag` | Whether this prediction generated an alert. |
| `alert_id` | Join key to alert and action fact tables. |
| `actual_recall_flag` | Outcome once known. Null before the outcome window closes. |
| `actual_recall_timestamp_utc` | Outcome timestamp when applicable. |
| `recall_quantity` | Quantity recalled when applicable. |
| `expected_loss_amount` | Optional model/business calculation for prioritized intervention. |
| `expected_value_amount` | Optional expected benefit net of intervention cost. |

### Implementation steps for PowerBI ingestion

1. Create certified SQL views or lakehouse tables for `fact_recall_prediction`, `fact_recall_alert`, `fact_user_action`, `fact_recall_outcome`, and core dimensions for date, loan, security, lender, borrower, desk, risk band, model version, and reason code.
2. Flatten or bridge nested model outputs. Probabilities should be stored in long form by horizon. Reason codes can be exposed either as ordered columns for simple tables or as a bridge table for multi-select analysis.
3. Normalize timestamps to UTC and provide local-market display columns only in dimensions or report logic.
4. Add schema contracts and data-quality checks that fail the BI publish if required columns, types, uniqueness, or date partitions are missing.
5. Partition large facts by `as_of_date` and configure incremental refresh so PowerBI only reloads new or recently changed partitions.
6. Certify DAX measures centrally in the semantic model. Report authors should reuse certified measures instead of redefining recall rate, alert volume, precision, or business-value logic.
7. Implement row-level security and object-level security before publishing to production workspaces.
8. Add reconciliation tiles to an admin page showing latest scoring timestamp, source row counts, PowerBI row counts, failed quality rows, and refresh status.
9. Validate representative user journeys with the desk, operations, model risk, and product before release.
10. Document report ownership, support escalation, refresh SLA, data lineage, semantic-model dependencies, and approved enhancement process.

### PowerBI acceptance criteria

- A certified semantic model exists and is owned by BI/data engineering.
- Operational pages can filter by date, horizon, desk, lender, security, risk band, model version, and data-quality status.
- Metrics reconcile to the prediction store within documented tolerance.
- Row-level security is tested for at least one user in each entitled role.
- The report includes data freshness, model version, feature-set version, and limitations text.
- Critical-risk loans can be exported or drilled through only by authorized users.
- Model-risk and monitoring pages distinguish open outcome windows from completed outcome windows.

See `docs/powerbi_analytics_guide.md` for the report design, DAX, calculated-column, and data-modeling guide.


## 7.9 Development Deliverables

- Source code
- CI/CD configuration
- Feature pipeline
- Training pipeline
- Scoring pipeline
- Model registry integration
- Prediction database
- API
- Dashboard
- Alert logic
- Unit tests
- Integration tests
- Documentation
- Prototype model card

## 7.10 Stage 3 Exit Criteria

- Code passes CI.
- Data tests pass.
- Training is reproducible.
- API contract is implemented.
- Model artifact is registered.
- All MVP features are implemented.
- Documentation is current.
- No critical security findings remain.

---

# Stage 4 — Verification, Validation, and Model Risk Review

## 8.1 Objectives

- Verify the software behaves as specified.
- Validate the model independently.
- Test business usefulness.
- Confirm controls, calibration, stability, and resilience.

## 8.2 Software Testing

### Unit Tests

- Label construction
- Date horizon logic
- Feature calculations
- Missing value handling
- Risk band assignment
- Reason code generation
- API schema
- Threshold logic

### Integration Tests

- Source to feature pipeline
- Feature store to scoring
- Scoring to prediction store
- Prediction store to API
- API to dashboard
- Alert generation
- Model registry loading

### Regression Tests

- Prediction consistency
- Schema stability
- Performance benchmark
- Known-case outputs
- Historical replay

### Resilience Tests

- Missing data source
- Delayed market data
- Partial source outage
- Corrupt model artifact
- API timeout
- Duplicate event
- Clock or timezone mismatch

## 8.3 Model Validation

Independent validation should assess:

- Conceptual soundness
- Label correctness
- Data quality
- Sampling
- Feature leakage
- Model assumptions
- Hyperparameter tuning
- Calibration
- Stability
- Bias
- Explainability
- Benchmarking
- Stress performance
- Limitations
- Implementation correctness

## 8.4 Backtesting

Perform rolling historical backtests.

Example:

- Train through 2024-12
- Test 2025-Q1
- Retrain through 2025-Q1
- Test 2025-Q2
- Continue sequentially

Assess:

- Performance over time
- Performance by lender
- Performance by security liquidity
- Performance by asset class
- Performance during stress
- Alert burden
- Economic value

## 8.5 Scenario Testing

Scenarios:

- Broad market selloff
- Index rebalance
- Proxy season
- Dividend season
- Large fund redemption
- Lender program change
- Borrow fee spike
- Settlement disruption
- Market holiday mismatch
- Data feed delay

## 8.6 Human-in-the-Loop Pilot

Run in shadow mode before production action.

Capture:

- Predictions
- User assessment
- Whether action was taken
- Whether recall occurred
- Whether intervention was useful
- False-alert reason
- Missed-event reason

## 8.7 Validation Deliverables

- Test report
- Independent validation report
- Benchmark comparison
- Backtest report
- Calibration report
- Stability report
- Fairness and conduct assessment
- Pilot results
- Remediation log
- Final model card
- Production readiness recommendation

## 8.8 Stage 4 Exit Criteria

- No unresolved critical defects.
- Independent validation approves or conditionally approves.
- Model exceeds baseline.
- Calibration meets agreed tolerance.
- Pilot demonstrates business value.
- Alert volume is operationally manageable.
- Limitations are documented.
- Required controls are implemented.

---

# Stage 5 — Deployment and Release

## 9.1 Objectives

- Deploy safely.
- Establish release, rollback, access, monitoring, and support.
- Transition ownership from development to production teams.

## 9.2 Release Strategy

Recommended sequence:

1. Development environment
2. Integration environment
3. User acceptance environment
4. Shadow production
5. Limited production pilot
6. Broader production release
7. Optimizer integration

## 9.3 Deployment Modes

### Daily Batch

Best initial production mode.

- Score before market open
- Re-score after major event updates
- Store all results
- Generate alerts after quality checks

### Intraday Event-Driven

Later phase.

Trigger on:

- New corporate action
- Material utilization change
- Large price move
- Lender activity change
- New recall message
- Inventory shock

## 9.4 Release Controls

- Versioned model artifact
- Versioned feature definitions
- Signed deployment package
- Automated smoke tests
- Approval workflow
- Canary or limited-user rollout
- Rollback command
- Feature flags
- Model kill switch
- Data freshness gate
- Prediction volume sanity check

## 9.5 Production Runbook

The runbook shall include:

- Service owner
- Support contacts
- Operating schedule
- Dependencies
- Health checks
- Expected volumes
- Common failure modes
- Restart procedure
- Rollback procedure
- Data delay procedure
- Escalation path
- Incident severity definitions
- Communication templates

## 9.6 Service-Level Objectives

Example targets:

| Measure | Target |
|---|---|
| Daily scoring completion | Before agreed business cut-off |
| Prediction availability | 99.5% during supported hours |
| API latency | Defined by platform standard |
| Data freshness | Within source-specific tolerance |
| Critical incident response | Per enterprise severity policy |
| Model rollback | Within approved recovery objective |

## 9.7 Deployment Deliverables

- Production pipeline
- Infrastructure as code
- Release checklist
- Rollback plan
- Access control matrix
- Support runbook
- User guide
- Training materials
- Production model card
- Go-live approval

## 9.8 Stage 5 Exit Criteria

- Production smoke tests pass.
- Monitoring is active.
- Support ownership is accepted.
- Rollback has been tested.
- Users are trained.
- Data quality gates are active.
- Go-live approval is documented.

---

# Stage 6 — Operations, Monitoring, and Continuous Improvement

## 10.1 Objectives

- Monitor technical, statistical, and business performance.
- Detect deterioration.
- Retrain safely.
- Learn from user actions and outcomes.
- Maintain governance and audit readiness.

## 10.2 Monitoring Layers

### Infrastructure Monitoring

- Pipeline completion
- Job failures
- Latency
- API availability
- CPU and memory
- Storage
- Dependency health

### Data Monitoring

- Record counts
- Missingness
- Freshness
- Schema changes
- Range violations
- Duplicate rates
- Join rates
- Entity coverage

### Feature Monitoring

- Distribution drift
- Null drift
- Out-of-range values
- Cardinality changes
- Correlation changes
- Feature availability

### Model Monitoring

- Prediction distribution
- Calibration
- PR-AUC
- Precision at top K
- Recall at top K
- Brier score
- Segment performance
- Regime performance
- Confidence distribution

### Business Monitoring

- Alerts generated
- Alerts reviewed
- Actions taken
- Recalls captured
- Avoided fails
- Emergency sourcing avoided
- Estimated cost savings
- User adoption
- Override rate

## 10.3 Drift Triggers

Trigger investigation when:

- PSI exceeds threshold
- Calibration error exceeds tolerance
- Precision at top K falls materially
- Alert volume changes unexpectedly
- Data freshness breaches SLA
- Segment performance diverges
- User override rate rises
- Business value declines

## 10.4 Retraining Policy

Retraining may be:

- Scheduled quarterly
- Triggered by drift
- Triggered by business process change
- Triggered by source data change
- Triggered by new asset class
- Triggered by performance decline

Retraining does not imply automatic promotion.

Every candidate model must:

- Reproduce training
- Pass validation
- Beat current champion
- Preserve calibration
- Pass stability checks
- Receive approval

## 10.5 Champion-Challenger Framework

Maintain:

- Champion model in production
- Challenger model in shadow mode
- Benchmark rules model
- Segment-level performance comparison

Promotion requires documented evidence.

## 10.6 Incident Management

Examples:

- Prediction job missed
- Source data delayed
- Model file corrupt
- Alert storm
- Material misclassification
- Unauthorized access
- Silent feature failure
- Incorrect model version

Each incident requires:

- Detection
- Containment
- Business communication
- Root-cause analysis
- Remediation
- Control improvement
- Closure approval

## 10.7 Continuous Improvement Backlog

Potential enhancements:

- Survival modeling
- Partial recall quantity prediction
- Graph features
- Sequence models
- Event-driven scoring
- Uncertainty quantification
- Personalized lender models
- Active learning
- Optimizer integration
- Automated inventory reservation
- Client-facing analytics
- Causal intervention analysis

## 10.8 Stage 6 Deliverables

- Monitoring dashboards
- Monthly performance report
- Drift report
- Retraining report
- Incident log
- Change log
- User feedback log
- Benefits realization report
- Annual model review
- Updated risk assessment

## 10.9 Stage 6 Exit Criteria

Stage 6 is continuous. The system may be retired when:

- Business value no longer justifies operation
- Data is no longer available
- Regulatory constraints change
- A replacement system is approved
- Performance cannot be remediated

---

# 11. Feature Specification

## 11.1 Loan-Level Features

- Loan age
- Current quantity
- Original quantity
- Quantity change
- Market value
- Fee or rebate
- Fee percentile
- Open versus term
- Days to maturity
- Prior partial return activity
- Loan utilization duration
- Recent amendments
- Collateral type
- Settlement location

## 11.2 Lender-Level Features

- Historical recall rate
- Recall rate by horizon
- Recall rate by security type
- Recall rate during proxy season
- Recall rate around dividends
- Average loan duration
- Recent recall velocity
- Concentration by fund
- Lender activity change
- Lender-specific seasonality

## 11.3 Security-Level Features

- Utilization
- Borrow fee
- Fee change
- Available lendable supply
- Short interest
- Days to cover
- Volatility
- Volume
- Bid-ask spread
- Return momentum
- Corporate actions
- Proxy record date proximity
- Dividend date proximity
- Index rebalance proximity
- Liquidity score
- Specialness indicator

## 11.4 Market and Regime Features

- Market volatility index
- Sector volatility
- Funding stress
- Market return
- Credit spread
- Liquidity regime
- Holiday calendar
- Quarter-end indicator
- Month-end indicator
- Year-end indicator
- Proxy season indicator

## 11.5 Operational Features

- Settlement history
- Fail rate
- Recall processing time
- Manual exception count
- Data quality flags
- Inventory replacement availability
- Current desk workload
- Message channel
- Cut-off proximity

---

# 12. Classification Design

## 12.1 Class Imbalance

Recalls may be rare relative to open loans.

Use:

- Class weighting
- Stratified time-aware sampling
- Downsampling only for training where justified
- Full-distribution validation
- PR-AUC rather than relying on ROC-AUC
- Precision-at-capacity metrics

Avoid synthetic oversampling unless its implications are carefully validated.

## 12.2 Calibration

Calibrate using:

- Platt scaling
- Isotonic regression
- Beta calibration

Evaluate calibration by:

- Reliability diagrams
- Brier score
- Expected calibration error
- Segment calibration

## 12.3 Risk Bands

Example:

| Risk Band | Probability | Intended Use |
|---|---:|---|
| Low | < 10% | No action |
| Moderate | 10–25% | Monitor |
| High | 25–50% | Review inventory |
| Critical | > 50% | Prepare intervention |

Thresholds must be tuned to operational capacity and economics.

## 12.4 Explainability

Provide:

- Global feature importance
- SHAP-based local reason codes
- Counterfactual explanation where appropriate
- Segment-level behavior
- Known limitations

Reason codes must be stable, understandable, and approved.

---

# 13. Integration with Optimization

The recall model should eventually influence:

- Inventory reservation
- Loan allocation
- Replacement sourcing
- Collateral substitution
- Pricing
- Client prioritization

## 13.1 Risk-Adjusted Allocation Objective

A simplified optimizer objective could include:

\[
\max \left(
\text{Expected Revenue}
- \lambda_1 \times \text{Recall Risk Cost}
- \lambda_2 \times \text{Fail Risk Cost}
- \lambda_3 \times \text{Transaction Cost}
\right)
\]

Where recall risk cost may be:

\[
\text{Recall Risk Cost}_i
=
P_i(\text{recall}) \times \text{Expected Replacement Cost}_i
\]

## 13.2 Guardrails

- Do not let ML probability override legal eligibility.
- Do not reserve inventory solely on uncalibrated predictions.
- Cap the economic penalty contribution.
- Require explainability for material allocation changes.
- Log model-driven optimizer decisions.
- Maintain human override.

---

# 14. Data Model

## 14.1 Prediction Table

| Field | Type |
|---|---|
| prediction_id | string |
| loan_id | string |
| as_of_timestamp | timestamp |
| model_version | string |
| feature_set_version | string |
| probability_1d | decimal |
| probability_3d | decimal |
| probability_5d | decimal |
| probability_10d | decimal |
| risk_band | string |
| reason_code_1 | string |
| reason_code_2 | string |
| reason_code_3 | string |
| data_quality_status | string |
| created_timestamp | timestamp |

## 14.2 Outcome Table

| Field | Type |
|---|---|
| loan_id | string |
| prediction_id | string |
| recall_flag | integer |
| recall_timestamp | timestamp |
| recall_quantity | decimal |
| full_recall_flag | integer |
| user_action | string |
| action_timestamp | timestamp |
| intervention_success | integer |
| estimated_value | decimal |
| notes | string |

---

# 15. Model Card Template

## Model Name

Loan Recall Prediction Model

## Intended Use

Predict near-term lender-initiated recall risk for open securities loans.

## Users

Desk, operations, inventory, product, optimization systems.

## Prohibited Use

- Automatic client communication without approval
- Legal interpretation
- Autonomous transaction termination
- Use outside approved markets or asset classes

## Training Data

Document:

- Date range
- Markets
- Asset classes
- Source systems
- Exclusions
- Known quality issues

## Performance

Report:

- Overall metrics
- Segment metrics
- Calibration
- Stress periods
- Confidence intervals

## Limitations

Examples:

- Cold-start lenders
- Sparse securities
- Unobserved fund flows
- Process changes
- Data latency
- Corporate action data quality

## Monitoring

Define:

- Frequency
- Thresholds
- Owners
- Escalation

---

# 16. Test Plan

## 16.1 Example Test Cases

| ID | Test | Expected Result |
|---|---|---|
| T-001 | Recall occurs within 1 day | 1d label = 1 |
| T-002 | Recall occurs after 1 day but within 3 days | 1d = 0, 3d = 1 |
| T-003 | Borrower return without recall | All recall labels = 0 |
| T-004 | Loan already under recall | Excluded |
| T-005 | Missing lender feature | Approved imputation |
| T-006 | Future timestamp present | Pipeline fails |
| T-007 | Prediction probability outside [0,1] | Pipeline fails |
| T-008 | Model artifact unavailable | Fallback procedure |
| T-009 | Data delayed | Alerts suppressed or flagged |
| T-010 | Duplicate recall event | Deduplicated |

---

# 17. Governance and Controls

## 17.1 Required Controls

- Data lineage
- Access control
- Segregation of duties
- Code review
- Model approval
- Version control
- Change management
- Audit logging
- Reproducibility
- Monitoring
- Annual review
- Retirement plan

## 17.2 Model Risk Tiering

Assess:

- Financial materiality
- Client impact
- Degree of automation
- Operational dependency
- Complexity
- Regulatory relevance

The model should initially be treated as decision support. Model risk classification may increase once integrated directly into allocation or pricing.

## 17.3 Fairness and Conduct

Although this is not a consumer credit model, assess whether model-driven actions could:

- Systematically disadvantage certain clients
- Create inconsistent service
- Encode relationship bias
- Cause inappropriate pricing behavior
- Conflict with contractual obligations

---

# 18. Delivery Roadmap

## Phase 0 — Feasibility

- Confirm label quality
- Build event taxonomy
- Produce baseline statistics
- Estimate economic value
- Approve MVP

## Phase 1 — Research Prototype

- Build point-in-time dataset
- Establish rules baseline
- Train logistic and boosting models
- Backtest
- Validate signal

## Phase 2 — Production MVP

- Build automated pipeline
- Register model
- Deploy daily scoring
- Launch dashboard
- Run shadow mode

## Phase 3 — Controlled Pilot

- Release to limited users
- Capture actions
- Measure operational value
- Tune thresholds
- Address false alerts

## Phase 4 — Production Scale

- Expand users and markets
- Formalize monitoring
- Integrate alerts
- Add optimizer inputs

## Phase 5 — Advanced Capabilities

- Survival modeling
- Quantity prediction
- Event-driven scoring
- Graph features
- Automated decision support

---

# 19. Milestones and Work Breakdown

| Milestone | Major Output |
|---|---|
| M1 | Requirements and label approved |
| M2 | Point-in-time training dataset complete |
| M3 | Baseline and candidate models evaluated |
| M4 | Architecture and API approved |
| M5 | MVP pipeline implemented |
| M6 | Independent validation complete |
| M7 | Shadow pilot launched |
| M8 | Production release approved |
| M9 | Optimizer integration complete |
| M10 | Benefits realization review |

---

# 20. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Label ambiguity | High | Formal event taxonomy |
| Leakage | High | Point-in-time joins and review |
| Rare events | High | PR metrics and cost-sensitive learning |
| User alert fatigue | High | Capacity-aware thresholds |
| Data latency | High | Freshness gates |
| Process change | Medium | Drift and change monitoring |
| Cold-start lenders | Medium | Hierarchical and fallback models |
| Overfitting | High | Rolling out-of-time tests |
| Miscalibration | High | Calibration and monitoring |
| Model used outside scope | High | Access and API controls |
| Poor adoption | High | Workflow co-design |
| Optimizer overreaction | High | Penalty caps and guardrails |

---

# 21. Handoff Checklist

## Product Handoff

- [ ] Business objective approved
- [ ] Users identified
- [ ] Intervention workflow documented
- [ ] Success metrics approved
- [ ] Scope and exclusions approved

## Data Handoff

- [ ] Source owners identified
- [ ] Data contracts signed
- [ ] Point-in-time rules documented
- [ ] Quality thresholds approved
- [ ] Lineage documented

## Model Handoff

- [ ] Label version recorded
- [ ] Feature set version recorded
- [ ] Model artifact registered
- [ ] Validation complete
- [ ] Model card approved

## Engineering Handoff

- [ ] Repository documented
- [ ] CI/CD active
- [ ] Infrastructure deployed
- [ ] Rollback tested
- [ ] Runbook approved

## Operations Handoff

- [ ] Support owner assigned
- [ ] Alert workflow agreed
- [ ] Escalation path tested
- [ ] User training complete
- [ ] SLAs documented

## Governance Handoff

- [ ] Model approval recorded
- [ ] Access controls reviewed
- [ ] Audit logs active
- [ ] Change process documented
- [ ] Annual review scheduled

---

# 22. Definition of Done

The project is not complete when the model achieves a high AUC. It is complete when:

- The recall event is consistently defined.
- The prediction dataset is point-in-time correct.
- The model is calibrated and stable.
- The model outperforms rules and historical heuristics.
- The system operates reliably.
- Alerts are actionable.
- Users adopt the workflow.
- Business value is measured.
- Risks and limitations are controlled.
- Ownership is transferred.
- Monitoring and retraining are operational.

---

# 23. Recommended MVP Decision

For the first release, implement:

- Daily batch scoring
- 1-day, 3-day, 5-day, and 10-day horizons
- Gradient-boosted tree model plus logistic benchmark
- Probability calibration
- Top-K and economic-value thresholding
- Loan-level reason codes
- Dashboard and API
- Shadow-mode pilot
- Manual user action capture
- Monthly model performance review

Do not initially implement:

- Deep learning
- Fully autonomous inventory actions
- Real-time scoring for all events
- Direct client-facing predictions
- Unbounded optimizer penalties

This narrower scope is more likely to generate measurable value and survive model-risk review.

---

# 24. Final Recommendation

Treat loan recall prediction as a combined **machine learning, workflow, and optimization-control problem**.

The strongest design is not:

> Predict whether a loan will be recalled.

It is:

> Predict recall risk early enough, explain why the risk increased, route the signal to the correct user or optimizer, recommend a proportionate intervention, record the action, and measure whether the intervention reduced cost or risk.

That framing turns the model from a research exercise into a production financing capability.

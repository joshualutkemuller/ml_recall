# PowerBI Analytics Guide for Loan Recall Prediction

## Purpose

This guide defines how to expose loan recall prediction outputs in PowerBI so business users can monitor risk, prioritize interventions, track model performance, and measure realized value. It complements the implementation handoff by translating model outputs into report pages, a governed semantic model, certified DAX measures, calculated columns, and operational release controls.

## Recommended PowerBI Pages

### 1. Executive Overview

Audience: executive sponsor, product owner, desk leads, operations leadership.

Recommended visuals:

- KPI cards for open loans, high-risk loans, critical-risk loans, alert volume, action completion rate, realized recalls, and estimated exposure.
- Trend lines for high-risk exposure and alert volume by as-of date.
- Stacked bars for risk-band distribution by horizon.
- Matrix by desk, region, asset class, and risk band.
- Data freshness, model version, and feature-set version callouts.

Primary questions:

- Is recall risk increasing or decreasing?
- Which desks, lenders, asset classes, or regions need attention?
- Are users acting on alerts within the expected SLA?

### 2. High-Risk Loan Workbench

Audience: desk users, inventory managers, and operations.

Recommended visuals:

- Ranked loan table with recall probability, risk band, expected loss, expected value, lender, security, quantity, fee, reason codes, and action status.
- Slicers for as-of date, horizon, desk, lender, security, borrower, risk band, and data-quality status.
- Drill-through to loan detail.
- Conditional formatting for critical rows, stale predictions, failed data quality, and overdue actions.

Primary questions:

- Which loans should be reviewed first?
- Why is each loan risky?
- Has someone already taken action?

### 3. Lender and Security Concentration

Audience: product, desk, inventory management, and risk.

Recommended visuals:

- Heat map of recall probability by lender and security sector.
- Top-N lenders by critical exposure.
- Top-N securities by predicted recall quantity or expected loss.
- Scatter plot of utilization ratio versus recall probability.
- Drill-through from lender/security to the affected loans.

Primary questions:

- Are risks concentrated in specific lenders, securities, sectors, or markets?
- Are upcoming corporate actions or market events driving clusters?

### 4. Alert and Action Management

Audience: operations, desk supervisors, and product owners.

Recommended visuals:

- Funnel from predictions to alerts, acknowledged alerts, actions taken, closed actions, and actual recalls.
- Aging buckets for unacknowledged and open alerts.
- Action distribution by action type and owner.
- SLA breach table.
- Suppression reason analysis.

Primary questions:

- Are alerts being handled quickly enough?
- Which actions are most common?
- Are duplicate or low-value alerts creating fatigue?

### 5. Model Performance and Outcomes

Audience: quant, data science, model risk, product, and governance.

Recommended visuals:

- Precision, recall, PR-AUC proxy tables, Brier score, and calibration by horizon where certified backend aggregates are available.
- Actual recall rate by probability decile.
- Lift chart by risk band or top-K bucket.
- Confusion matrix controlled by selected threshold.
- Outcome-window completion indicators.

Primary questions:

- Are predicted probabilities calibrated?
- Does the model outperform baseline rules?
- Are top-ranked loans actually recalling at higher rates?

### 6. Data Quality and Refresh Monitoring

Audience: BI support, data engineering, production support, and model operations.

Recommended visuals:

- Latest source scoring timestamp versus latest PowerBI refresh timestamp.
- Source row counts versus semantic-model row counts.
- Failed and warning data-quality rows by reason.
- Missing dimension-key counts.
- Model version and feature-set version adoption trend.

Primary questions:

- Is the report fresh, complete, and trustworthy?
- Are quality issues isolated to a source, date, desk, or model version?

## Recommended Data Model

Use a star schema. Avoid report logic that joins directly across raw operational tables.

### Fact tables

- `fact_recall_prediction`: one row per loan, as-of date or timestamp, and horizon.
- `fact_recall_alert`: one row per generated alert.
- `fact_user_action`: one row per action, status update, or workflow event.
- `fact_recall_outcome`: one row per recall outcome event or outcome-window closure.
- `fact_model_metric_daily`: daily model metrics by horizon, model version, and segment, preferably pre-aggregated outside PowerBI.
- `fact_data_quality_check`: one row per quality check execution, date, source, and status.

### Dimensions

- `dim_date`: reporting calendar, business-day flags, market holidays, week, month, quarter, and year.
- `dim_loan`: loan attributes that are safe for reporting.
- `dim_security`: ticker, issuer, asset class, sector, country, exchange, liquidity bucket, and corporate-action flags.
- `dim_lender`: lender, fund, client segment, region, and row-level security attributes.
- `dim_borrower`: borrower, desk, business unit, and region.
- `dim_risk_band`: risk band label, ordinal sort order, display color, and threshold version.
- `dim_model_version`: model version, feature-set version, calibration version, approval status, and deployment dates.
- `dim_reason_code`: reason code, user-facing label, description, and approved display flag.
- `dim_action_type`: action taxonomy and SLA target.

### Relationships

- Relate facts to dimensions using single-direction filtering from dimensions to facts.
- Use `dim_date[Date]` to filter `fact_recall_prediction[as_of_date]`; create inactive relationships for alert, action, and outcome dates and activate them in measures when needed.
- Keep horizon as either a small dimension or a column on prediction and metric facts.
- Use a bridge table such as `bridge_prediction_reason_code` if analysts need reason-code slicing across multiple ordered reason-code fields.
- Prefer surrogate integer keys for large Import models and stable natural keys for DirectQuery only where the source system guarantees uniqueness.

## Certified DAX Measures

The following measures should be reviewed, named consistently, and certified in the shared semantic model. Table and column names may be adapted to the implemented schema.

```DAX
Open Loan Count =
DISTINCTCOUNT ( fact_recall_prediction[loan_id] )
```

```DAX
Prediction Count =
COUNTROWS ( fact_recall_prediction )
```

```DAX
High Risk Loan Count =
CALCULATE (
    DISTINCTCOUNT ( fact_recall_prediction[loan_id] ),
    fact_recall_prediction[risk_band] IN { "HIGH", "CRITICAL" }
)
```

```DAX
Critical Risk Loan Count =
CALCULATE (
    DISTINCTCOUNT ( fact_recall_prediction[loan_id] ),
    fact_recall_prediction[risk_band] = "CRITICAL"
)
```

```DAX
Average Recall Probability =
AVERAGE ( fact_recall_prediction[recall_probability] )
```

```DAX
Probability Weighted Exposure =
SUMX (
    fact_recall_prediction,
    fact_recall_prediction[recall_probability]
        * fact_recall_prediction[market_value_amount]
)
```

```DAX
Expected Loss Amount =
SUMX (
    fact_recall_prediction,
    fact_recall_prediction[recall_probability]
        * fact_recall_prediction[estimated_loss_given_recall_amount]
)
```

```DAX
Expected Net Benefit =
SUM ( fact_recall_prediction[expected_value_amount] )
```

```DAX
Alert Count =
COUNTROWS ( fact_recall_alert )
```

```DAX
Alerted Loan Count =
DISTINCTCOUNT ( fact_recall_alert[loan_id] )
```

```DAX
Action Completion Rate =
DIVIDE (
    CALCULATE ( COUNTROWS ( fact_user_action ), fact_user_action[action_status] = "CLOSED" ),
    COUNTROWS ( fact_user_action )
)
```

```DAX
SLA Breach Count =
CALCULATE (
    COUNTROWS ( fact_user_action ),
    fact_user_action[sla_breached_flag] = TRUE ()
)
```

```DAX
Actual Recall Count =
CALCULATE (
    COUNTROWS ( fact_recall_outcome ),
    fact_recall_outcome[actual_recall_flag] = TRUE ()
)
```

```DAX
Outcome Eligible Prediction Count =
CALCULATE (
    COUNTROWS ( fact_recall_prediction ),
    fact_recall_prediction[outcome_window_closed_flag] = TRUE ()
)
```

```DAX
Observed Recall Rate =
DIVIDE ( [Actual Recall Count], [Outcome Eligible Prediction Count] )
```

```DAX
Precision at Alerted =
DIVIDE (
    CALCULATE (
        COUNTROWS ( fact_recall_prediction ),
        fact_recall_prediction[alert_generated_flag] = TRUE (),
        fact_recall_prediction[actual_recall_flag] = TRUE (),
        fact_recall_prediction[outcome_window_closed_flag] = TRUE ()
    ),
    CALCULATE (
        COUNTROWS ( fact_recall_prediction ),
        fact_recall_prediction[alert_generated_flag] = TRUE (),
        fact_recall_prediction[outcome_window_closed_flag] = TRUE ()
    )
)
```

```DAX
Recall Captured by Alerts =
DIVIDE (
    CALCULATE (
        COUNTROWS ( fact_recall_prediction ),
        fact_recall_prediction[alert_generated_flag] = TRUE (),
        fact_recall_prediction[actual_recall_flag] = TRUE (),
        fact_recall_prediction[outcome_window_closed_flag] = TRUE ()
    ),
    CALCULATE (
        COUNTROWS ( fact_recall_prediction ),
        fact_recall_prediction[actual_recall_flag] = TRUE (),
        fact_recall_prediction[outcome_window_closed_flag] = TRUE ()
    )
)
```

```DAX
Latest Prediction Timestamp =
MAX ( fact_recall_prediction[as_of_timestamp_utc] )
```

```DAX
Latest Refresh Timestamp =
MAX ( fact_data_quality_check[powerbi_refresh_timestamp_utc] )
```

```DAX
Failed Quality Row Count =
CALCULATE (
    COUNTROWS ( fact_recall_prediction ),
    fact_recall_prediction[data_quality_status] = "FAIL"
)
```

```DAX
Stale Prediction Flag =
VAR HoursSincePrediction =
    DATEDIFF ( [Latest Prediction Timestamp], UTCNOW (), HOUR )
RETURN
    IF ( HoursSincePrediction > 24, 1, 0 )
```

## Calculated Column Suggestions

Prefer upstream SQL or lakehouse transformations for high-volume or security-sensitive logic. Use PowerBI calculated columns only for lightweight display and sorting logic.

Recommended calculated columns:

```DAX
Risk Band Sort Order =
SWITCH (
    fact_recall_prediction[risk_band],
    "LOW", 1,
    "MODERATE", 2,
    "HIGH", 3,
    "CRITICAL", 4,
    99
)
```

```DAX
Probability Bucket =
SWITCH (
    TRUE (),
    fact_recall_prediction[recall_probability] < 0.05, "00-05%",
    fact_recall_prediction[recall_probability] < 0.10, "05-10%",
    fact_recall_prediction[recall_probability] < 0.25, "10-25%",
    fact_recall_prediction[recall_probability] < 0.50, "25-50%",
    "50%+"
)
```

```DAX
Outcome Status =
SWITCH (
    TRUE (),
    fact_recall_prediction[outcome_window_closed_flag] = FALSE (), "Window Open",
    fact_recall_prediction[actual_recall_flag] = TRUE (), "Recalled",
    "No Recall"
)
```

```DAX
Alert Review Status =
SWITCH (
    TRUE (),
    fact_recall_prediction[alert_generated_flag] = FALSE (), "No Alert",
    ISBLANK ( fact_recall_prediction[action_status] ), "Needs Review",
    fact_recall_prediction[action_status]
)
```

```DAX
Reason Code Display =
fact_recall_prediction[reason_code_1] & " | "
    & fact_recall_prediction[reason_code_2] & " | "
    & fact_recall_prediction[reason_code_3]
```

Recommended upstream columns instead of PowerBI calculated columns:

- `outcome_window_closed_flag`, because it depends on business calendars and horizon logic.
- `business_days_since_alert`, because holiday calendars should be governed centrally.
- `expected_loss_amount` and `expected_value_amount`, because finance-approved formulas should be versioned and tested upstream.
- `risk_score_rank`, because ranking should be reproducible across report refreshes.
- `row_security_entitlement_key`, because security should not depend on report-author logic.

## Data Modeling and Performance Guidance

- Use Import mode for curated daily snapshots when latency requirements permit; use DirectQuery or composite models only when freshness materially changes user decisions.
- Configure incremental refresh on `as_of_date` and keep a refresh policy aligned to the model scoring cadence.
- Use aggregation tables for executive and trend pages when prediction history is large.
- Disable Auto Date/Time and rely on `dim_date`.
- Hide technical keys and raw columns that users should not analyze directly.
- Assign sort-by columns for risk band, horizon, probability bucket, and action status.
- Use certified perspectives or field folders for Executive, Operations, Model Risk, and Data Quality audiences.
- Keep probability measures numeric and format them as percentages in the semantic model.
- Store monetary fields in a single reporting currency or provide an approved currency conversion dimension.
- Precompute model metrics that require complex ranking, deciles, calibration curves, or backtest windows outside PowerBI when row volume is material.

## Security and Governance

- Implement row-level security by desk, region, client, lender, borrower, or portfolio as required by data-governance policy.
- Use object-level security to hide sensitive borrower, client, lender, fee, or explainability fields from unauthorized users.
- Separate production, validation, and development workspaces.
- Certify the production semantic model and prevent unmanaged copies from becoming official management reporting.
- Display model limitations, approved use, data freshness, model version, and contact owner on the report landing page.
- Audit report access, exports, subscriptions, and drill-through to loan-level details.

## Release Checklist

- Prediction facts reconcile to the prediction store for row count, date range, model version, and horizon distribution.
- Incremental refresh succeeds within the agreed SLA.
- Row-level security is tested with representative users from each role.
- Executive, operations, and model-risk pages have been reviewed by their intended users.
- Certified DAX measures have named owners and definitions.
- Data-quality warning and failure states are visible to admins and do not silently pollute operational pages.
- Export permissions match the sensitivity of loan-level details.
- Support runbook includes refresh failures, data-source delays, semantic-model deployment, and report rollback.

# NYC Accident Analysis Plan

Primary dataset: `NYC_Collisions_Official_full.csv` (official multi-year source, 2012-07-01 to 2026-03-03).

## COVID-19 Timeline Context for NYC Analysis (2020)

This timeline is included so 2020 crash-pattern changes can be analyzed against major policy and restriction shifts.

| Date | COVID / Policy Change in NYC | Why It Matters for Accident Analysis |
|---|---|---|
| March 16, 2020 | NYC public schools closed. | Commute and school-trip traffic patterns changed. |
| March 17, 2020 | Theaters, concert venues, and nightclubs shut; restaurants restricted to take-out/delivery; gyms closed. | Nightlife and leisure mobility dropped sharply; crash mix by time likely changed. |
| March 20, 2020 (announced) | NY State PAUSE order announced. | Start of strict non-essential activity restrictions. |
| March 22, 2020 at 8:00 PM | PAUSE order took effect (non-essential business closures + gathering restrictions). | Strong structural break point for traffic volume and severity analysis. |
| June 8, 2020 | NYC entered Reopening Phase 1. | Beginning of reopening regime; exposure patterns started to recover. |
| June 22, 2020 | NYC entered Reopening Phase 2. | Additional activity returned (for example retail/office/outdoor dining). |
| July 6, 2020 | NYC entered Reopening Phase 3, excluding indoor dining. | Partial reopening with continued indoor restrictions affects evening mobility risk. |
| July 20, 2020 | NYC entered Reopening Phase 4 (with some indoor activity limitations). | Another regime shift for trend and impact segmentation. |

## 2020 CSV Task Table (Single-Year Snapshot Plan)

Dataset focus for this section:
- `NYC Accidents 2020.csv`
- Effective coverage: `2020-01-01` to `2020-08-29` (single-year partial window)

| Step | Task Name | Description | Output | Visualizations / Graphs Produced |
|---:|---|---|---|---|
| A1 | 2020 Snapshot QA | Validate row count, date range, unique collision IDs, and schema consistency. | Snapshot quality summary report | QA summary bars |
| A2 | 2020 Missingness Audit | Profile missing fields in 2020 subset and flag high-impact columns. | Missing-data assessment table and notes | Missingness heatmap |
| A3 | 2020 Core Feature Build | Build `hour`, `day_of_week`, `is_weekend`, and lockdown phase labels for 2020 period. | 2020 engineered feature dataset | Feature coverage chart |
| A4 | 2020 Lockdown Phase Segmentation | Segment data into pre-PAUSE, PAUSE, and reopening phases for Jan-Aug 2020. | 2020 phase segmentation summary | Phase timeline chart |
| A5 | 2020 Baseline KPIs | Compute collisions, injuries, and fatalities by day, week, and month in the 2020 window. | 2020 KPI summary table | Time-trend line charts |
| A6 | 2020 Frequency vs Severity | Compare rush-hour vs non-rush-hour frequency and severity patterns. | Frequency-severity comparison summary | Hourly dual-axis risk chart |
| A7 | 2020 Weekend vs Weekday Factors | Evaluate Alcohol vs Inattention and other factor shifts by day type. | Day-type factor comparison table | Stacked factor bars |
| A8 | 2020 Vehicle Severity | Analyze vehicle-type links to injury/fatal outcomes in subset. | Vehicle severity risk summary | Vehicle severity comparison plot |
| A9 | 2020 Lethality Matrix | Build 2020-specific lethality matrix for pedestrian/cyclist harm. | 2020 lethality ranking table | Lethality heatmap |
| A10 | 2020 Hotspot Mapping | Build hotspot map for Jan-Aug 2020 collisions and severe events. | 2020 hotspot summary table | 2020 hotspot map |
| A11 | 2020 Spring DST Check | Analyze spring-forward window effects (available in this date range). | Spring DST impact note | Event-window trend plot |
| A12 | 2020 Fall DST Limitation Note | Document that fall-back DST is not available in current 2020 subset window. | Scope limitation note | None |
| A13 | 2020 Severity Model | Train 2020-only logistic baseline for `any_injury` and `fatal_collision`. | 2020 model performance summary | ROC, PR, and calibration curves |
| A14 | 2020 Policy Signal Summary | Summarize how March-July 2020 COVID policy shifts align with crash changes. | 2020 policy-signal interpretation report | Policy phase impact chart |
| A16 | 2020 Source-Map Visualization Suite | Build direct map visualizations from raw 2020 source coordinates to reveal road imprint and density patterns. | 2020 map visualization package + map QA summary | Road-imprint scatter map, density hexbin map, injury hotspot overlay map, hourly map triptych |

## Master Multi-Year Task Table (Official Full Dataset)

| Step | Coverage Source | Task Name | Description | Output | Visualizations / Graphs Produced |
|---:|---|---|---|---|---|
| 1 | Basic | Run Configuration Setup | Define paths, random seed, run ID, and execution metadata for reproducibility. | Run configuration spec | None |
| 2 | Basic | Official Data Ingestion | Load official CSV and convert to canonical analysis format. | Canonical analysis dataset | None |
| 3 | Basic | Schema Validation | Validate 29 columns, data types, and required-field rules. | Schema validation report | Data-quality summary bar chart |
| 4 | Basic | Row-Level Data QA | Detect invalid date/time entries, duplicate IDs, and missing-value patterns. | Data quality report | Missingness heatmap, QA issue-count plot |
| 5 | PDF Methodology | Vehicle Category Standardization | Normalize high-cardinality vehicle labels (for example SUV variants). | Vehicle normalization rules and summary | Top vehicle-category distribution chart |
| 7 | PDF Methodology | Time/Policy Feature Engineering | Create `is_weekend`, `time_block`, `pandemic_era`, and `dst_transition`. | Time and policy feature dataset | Feature coverage summary chart |
| 9 | Basic | Final Feature Table Build | Produce cleaned feature table for all downstream analyses and models. | Final modeling-ready feature table | None |
| 10 | Basic | Baseline KPI Generation | Compute core KPIs by year, month, day, hour, and borough. | Baseline KPI table | Multi-year trend lines, hourly risk plot |
| 11 | PDF Research I | Vehicle vs Casualty Severity Analysis | Evaluate how vehicle type relates to injury/fatality severity outcomes. | Vehicle-severity findings report | Vehicle-type severity comparison chart |
| 12 | PDF Deliverable 2 | Lethality Matrix Construction | Rank vehicle risk to pedestrians and cyclists using harm metrics. | Lethality matrix table | Lethality matrix heatmap |
| 13 | PDF Research II | Pandemic Aftermath Analysis | Compare pre-2020 vs 2020-2021 vs 2022+ trends (including unsafe speed factors). | Pandemic aftermath findings report | Era comparison plot, factor trend chart |
| 14 | PDF Deliverable 3 | Pandemic Impact Report | Build full pandemic impact narrative with pre-pandemic baseline comparison. | Pandemic impact report | Pandemic period KPI dashboard plots |
| 15 | PDF Research III | Frequency vs Severity Analysis | Compare crash frequency and severity by rush-hour vs non-rush-hour windows. | Frequency-severity findings summary | Hour-of-day frequency-severity dual-axis chart |
| 17 | PDF Research IV | Weekend vs Weekday Factors | Compare factor patterns (Alcohol vs Inattention) between weekday and weekend. | Weekday-weekend factor report | Factor share stacked bars |
| 21 | PDF Deliverable 1 | Safety Hotspot Map | Build hotspot mapping by time, vehicle mix, and severity level. | Hotspot mapping output package | Interactive hotspot map |
| 22 | Deep | Spatial Hotspot Clustering | Detect clusters using spatial algorithms (DBSCAN/HDBSCAN style). | Spatial cluster table and notes | Cluster map with boundaries |
| 27 | Deep | Logistic Severity Baseline | Train logistic baseline with class weighting and evaluate discrimination. | Logistic model performance summary | ROC, PR, and calibration curves |
| 32 | Deep | Source-Map Visualization Suite (Full Data) | Build map visualizations directly from official full CSV coordinates for spatial pattern validity and presentation quality. | Full-data map visualization package + map QA summary | Road-imprint scatter map, density hexbin map, injury hotspot overlay map, hourly map triptych |

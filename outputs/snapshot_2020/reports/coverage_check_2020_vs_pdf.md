# Coverage Check: 2020 CSV vs Proposal PDF

Scope checked:
- Proposal: `docs/ECE 143 Project Proposal.pdf`
- Data: `data/NYC Accidents 2020.csv` (2020-01-01 to 2020-08-29)

## A-Table Coverage (2020 Snapshot Tasks)

| Task | Status | Evidence |
|---|---|---|
| A1 Snapshot QA | Done | `reports/data_quality_report.md`, `reports/canonical_dataset_report.md` |
| A2 Missingness audit | Done | `tables/missingness_summary.csv`, `visualizations/snapshot_2020/core/top_missing_columns.png` |
| A3 Core features (`hour`, `day_of_week`, `is_weekend`, phase) | Done | `tables/feature_table.csv`, `reports/feature_engineering_report.md` |
| A4 Lockdown phase segmentation | Done | `tables/phase_segmentation_summary_2020.csv`, `reports/phase_segmentation_report_2020.md`, `visualizations/snapshot_2020/core/phase_collision_count_2020.png` |
| A5 Baseline KPIs | Done | `tables/monthly_kpi.csv`, `tables/weekly_kpi_2020.csv`, `tables/hourly_kpi.csv`, trend charts |
| A6 Frequency vs severity | Done | `tables/hourly_frequency_severity_2020.csv`, `visualizations/snapshot_2020/core/hourly_fatal_rate_2020.png` |
| A7 Weekend vs weekday factors | Done | `tables/weekend_weekday_factors_2020.csv`, `visualizations/snapshot_2020/core/weekend_factor_share_2020.png` |
| A8 Vehicle severity | Done | `tables/vehicle_severity_summary_2020.csv`, `visualizations/snapshot_2020/core/vehicle_killed_per_10k_2020.png` |
| A9 Lethality matrix | Done | `tables/lethality_matrix_2020.csv`, `visualizations/snapshot_2020/heatmaps/lethality_matrix_2020_heatmap.png` |
| A10 Hotspot mapping | Done | `tables/hotspot_summary_2020.csv`, `visualizations/snapshot_2020/clusters/hotspot_grid_centers_2020.png` |
| A11 Spring DST check | Done | `reports/dst_spring_check_2020.md` |
| A12 Fall DST limitation note | Done | `reports/dst_fall_limitation_2020.md` |
| A13 Severity model | Done earlier (separate step) | `reports/severity_model_report.md`, model charts in `visualizations/snapshot_2020/models/` |
| A14 COVID lockdown impact signal | Done | `reports/covid_policy_signal_2020.md`, phase plots |
| A15 2020 vs multi-year bridge | Not done yet | No dedicated bridge report generated in this pass |

## PDF Requirement Coverage (for 2020 CSV scope)

| PDF Item | Status | Notes |
|---|---|---|
| Vehicle Type vs Casualty Severity | Done | Covered by A8 + lethality outputs. |
| Frequency vs Severity (rush vs non-rush) | Done | Covered by A6 outputs. |
| Weekend vs Weekday factors | Done | Covered by A7 outputs. |
| DST Spring analysis | Done | Covered by A11 outputs. |
| DST Fall analysis | Not possible from this file window | 2020 CSV ends on 2020-08-29; fall DST needs Nov 2020 rows. |
| Safety Hotspot Map deliverable | Done (static analysis map output) | Grid hotspot outputs generated. |
| Lethality Matrix deliverable | Done | Matrix table + heatmap generated. |
| Pandemic impact within 2020 phases | Done | Covered by A4 + A14 outputs. |
| Pandemic Aftermath (Pre-2020 vs 2020-2021 vs 2022+) | Not possible using only 2020 snapshot file | Requires multi-year data. |
| Policy recommendations deliverable | Not included in this first-step pass | Can be added after final interpretation phase. |

## Conclusion

- Your concern was correct: previously, the 2020 snapshot outputs were incomplete for the A-table/PDF-first-step scope.
- After this pass, the core 2020 basic analysis, graphs, heatmaps, and COVID-lockdown-impact outputs are now in place.
- Remaining 2020-scope gap: A15 bridge report.
- Remaining PDF items that require broader data or later stage: multi-year aftermath comparison and policy recommendation package.

# NYC Motor Vehicle Collision Analysis

A data science pipeline for analyzing NYC motor vehicle collision data sourced from the NYPD Motor Vehicle Collisions dataset. The project performs end-to-end analysis including data ingestion, quality checks, feature engineering, exploratory analysis, severity modeling, and spatial clustering — across both a single-year 2020 snapshot and a full multi-year dataset spanning 2012–2026.

---

## Table of Contents

- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Datasets](#datasets)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Third-Party Dependencies](#third-party-dependencies)
- [Outputs](#outputs)

---

## Project Overview

The pipeline supports two run modes:

| Mode | Config File | Dataset | Focus |
|---|---|---|---|
| `snapshot_2020` | `configs/snapshot_2020.json` | `NYC Accidents 2020.csv` | Jan–Aug 2020, COVID-19 policy impact |
| `full_multiyear` | `configs/full_multiyear.json` | `NYC_Collisions_Official_full.csv` | 2012–2026, long-term trend analysis |

---

## File Structure

```
Accident-Data-Analysis/
│
├── configs/                          # Run configuration files
│   ├── run_config.json               # Generic template config
│   ├── snapshot_2020.json            # Config for the 2020 single-year run
│   └── full_multiyear.json           # Config for the full multi-year run
│
├── data/                             # Raw input datasets (see Datasets section)
│   ├── NYC Accidents 2020.csv        # 2020 snapshot (Jan–Aug 2020)
│   └── NYC_Collisions_Official_full.csv  # Full official dataset (2012–2026)
│
├── docs/
│   └── ECE 143 Project Proposal.pdf  # Original academic project proposal
│
├── scripts/                          # All pipeline stage scripts
│   ├── setup/
│   │   ├── init_project.py           # One-time project initialization
│   │   └── run_pipeline.py           # Main entry point — runs all stages sequentially
│   │
│   ├── data_ingestion/
│   │   └── build_canonical_dataset.py    # Loads and normalizes the raw CSV
│   │
│   ├── data_quality/
│   │   └── run_data_quality_checks.py    # Validates schema, detects missing values and duplicates
│   │
│   ├── feature_engineering/
│   │   └── build_feature_table.py        # Derives analytical features (hour, is_weekend, pandemic_era, etc.)
│   │
│   ├── exploratory_analysis/
│   │   ├── create_core_outputs.py             # Core KPI tables and charts
│   │   ├── create_source_map_visualizations.py  # GPS-based map visualizations
│   │   ├── run_full_multiyear_deep_analysis.py  # Multi-year deep analysis
│   │   └── run_snapshot_2020_completion.py      # 2020-specific COVID phase analysis
│   │
│   ├── severity_models/
│   │   └── train_baseline_severity_model.py  # Logistic regression for injury/fatality prediction
│   │
│   ├── clustering_analysis/
│   │   └── run_spatial_clustering.py         # MiniBatchKMeans spatial hotspot clustering
│   │
│   └── validation_tests/
│       └── run_validation_checks.py          # Cross-checks output consistency
│
├── utils/                            # Shared utility modules
│   ├── __init__.py
│   ├── config_utils.py               # Config loading and CLI argument parsing
│   ├── io_utils.py                   # File I/O helpers (CSV, JSON, Markdown writers)
│   ├── log_utils.py                  # Logging setup and step logger
│   ├── path_utils.py                 # Directory creation helpers
│   ├── plot_utils.py                 # Chart generation wrappers (bar, line, heatmap, scatter, hexbin)
│   ├── stats_utils.py                # Statistical helpers (safe division, aggregations)
│   └── time_utils.py                 # Date/time parsing and pandemic phase labeling
│
├── outputs/                          # Generated reports and data tables (created at runtime)
│   ├── full_multiyear/
│   │   ├── reports/                  # Markdown analysis reports
│   │   └── tables/                   # CSV output tables (KPIs, clusters, model metrics)
│   └── snapshot_2020/
│       ├── reports/
│       └── tables/
│
├── visualizations/                   # Generated charts and maps (created at runtime)
│   ├── full_multiyear/
│   │   ├── core/                     # Trend lines, bar charts, factor plots
│   │   ├── heatmaps/                 # Hour×day and lethality matrix heatmaps
│   │   ├── maps/                     # Spatial density and hotspot maps
│   │   ├── clusters/                 # Cluster scatter and hexbin plots
│   │   └── models/                   # ROC, precision-recall, calibration curves
│   └── snapshot_2020/
│       ├── core/
│       ├── heatmaps/
│       ├── maps/
│       ├── clusters/
│       └── models/
│
└── internal.md                       # Internal task planning document
```

---

## Datasets

The raw data files are stored under `data/` and are tracked via Git LFS. You will need Git LFS installed to pull the actual CSV contents:

```bash
git lfs install
git lfs pull
```

| File | Rows (approx.) | Date Range | Notes |
|---|---|---|---|
| `NYC Accidents 2020.csv` | ~120,000 | Jan 1 – Aug 29, 2020 | Partial-year snapshot |
| `NYC_Collisions_Official_full.csv` | ~2,246,000 | Jul 1, 2012 – Mar 3, 2026 | Full official dataset |

Both files use the standard NYPD Motor Vehicle Collisions schema with 29 columns including crash date/time, borough, ZIP, GPS coordinates, injury/fatality counts, contributing factors, and vehicle type codes.

---

## How to Run

### Prerequisites

Ensure you have Python 3.9+ installed. Install the one required third-party dependency:

```bash
pip install matplotlib
```

### Update Config Paths

Before running, open the config file you intend to use and update all file paths to match your local environment. The configs currently contain absolute paths from the original development machine:

```json
{
  "source_csv": "/your/local/path/data/NYC_Collisions_Official_full.csv",
  "canonical_csv": "/your/local/path/outputs/full_multiyear/tables/canonical_dataset.csv",
  ...
}
```

### Run the Full Pipeline

To run all 7 stages end-to-end for a given config:

```bash
# Full multi-year analysis (2012–2026)
python scripts/setup/run_pipeline.py --config configs/full_multiyear.json

# 2020 snapshot analysis (COVID phase segmentation)
python scripts/setup/run_pipeline.py --config configs/snapshot_2020.json
```

### Run Individual Stages

Each script can also be run independently:

```bash
python scripts/data_ingestion/build_canonical_dataset.py      --config configs/full_multiyear.json
python scripts/data_quality/run_data_quality_checks.py        --config configs/full_multiyear.json
python scripts/feature_engineering/build_feature_table.py     --config configs/full_multiyear.json
python scripts/exploratory_analysis/create_core_outputs.py    --config configs/full_multiyear.json
python scripts/severity_models/train_baseline_severity_model.py --config configs/full_multiyear.json
python scripts/clustering_analysis/run_spatial_clustering.py  --config configs/full_multiyear.json
python scripts/validation_tests/run_validation_checks.py      --config configs/full_multiyear.json
```

Stages must be run in the above order as each stage depends on outputs from the previous one.

---

## Configuration

Each JSON config file controls the full run. Key fields:

| Field | Description |
|---|---|
| `dataset_name` | Label used for output folder naming (`snapshot_2020` or `full_multiyear`) |
| `source_csv` | Path to the raw input CSV |
| `canonical_csv` | Output path for the cleaned canonical dataset |
| `feature_csv` | Output path for the engineered feature table |
| `reports_dir` | Directory for markdown report outputs |
| `tables_dir` | Directory for CSV table outputs |
| `visualizations_dir` | Directory for PNG chart outputs |
| `random_seed` | Seed for reproducible model training and sampling |
| `max_rows_for_model` | Row cap for severity model training (default: 250,000) |
| `cluster_sample_limit` | Row cap for spatial clustering |
| `plot_sample_limit` | Row cap for scatter plot rendering |
| `n_spatial_clusters` | Number of clusters for MiniBatchKMeans (45 for 2020, 140 for full) |
| `cluster_top_n` | Number of top clusters to report |

---

## Pipeline Stages

| # | Script | What It Does |
|---|---|---|
| 1 | `build_canonical_dataset.py` | Loads raw CSV, normalizes column names and aliases, parses dates, writes a clean canonical CSV |
| 2 | `run_data_quality_checks.py` | Validates schema, checks for duplicate collision IDs, profiles missing values, generates QA report |
| 3 | `build_feature_table.py` | Derives `hour`, `day_of_week`, `is_weekend`, `time_block`, `pandemic_era`, and `dst_transition` columns |
| 4 | `create_core_outputs.py` | Computes KPIs by year/month/day/hour/borough; generates trend charts, factor plots, and vehicle type distributions |
| 5 | `train_baseline_severity_model.py` | Trains a logistic regression model (from scratch) to predict `any_injury` and `fatal_collision`; outputs ROC, PR, and calibration curves |
| 6 | `run_spatial_clustering.py` | Clusters GPS coordinates using MiniBatchKMeans (from scratch); outputs hotspot maps and ranked cluster tables |
| 7 | `run_validation_checks.py` | Cross-validates output totals against canonical dataset for consistency |

---

## Third-Party Dependencies

This project is intentionally lean. The only third-party library required is:

| Library | Version | Usage |
|---|---|---|
| `matplotlib` | ≥ 3.5 | All chart and map visualizations (bar charts, line plots, heatmaps, scatter maps, hexbin density maps) |

All other functionality — including the logistic regression model, spatial clustering (MiniBatchKMeans), date parsing, statistics, CSV I/O, and logging — is implemented using Python's standard library only.

**Standard library modules used:** `argparse`, `array`, `collections`, `csv`, `dataclasses`, `datetime`, `hashlib`, `json`, `logging`, `math`, `pathlib`, `random`, `statistics`, `subprocess`, `sys`, `typing`

To install the single dependency:

```bash
pip install matplotlib
```

Or pin to a specific version:

```bash
pip install matplotlib==3.9.0
```

---

## Outputs

After a successful run, the following are generated:

**Reports** (Markdown, under `outputs/<run>/reports/`):
- `canonical_dataset_report.md` — ingestion summary
- `data_quality_report.md` — missingness and QA findings
- `feature_engineering_report.md` — derived feature coverage
- `exploratory_analysis_report.md` — EDA narrative
- `severity_model_report.md` — model performance summary
- `spatial_clustering_report.md` — cluster findings
- `validation_checks_report.md` — consistency check results
- *(2020 only)* `covid_policy_signal_2020.md`, `phase_segmentation_report_2020.md`

**Tables** (CSV, under `outputs/<run>/tables/`):
- `canonical_dataset.csv`, `feature_table.csv`
- `hourly_kpi.csv`, `monthly_kpi.csv`, `yearly_kpi_multiyear.csv`
- `borough_collisions.csv`, `top_factors.csv`, `top_vehicle_types.csv`
- `lethality_matrix_*.csv`, `vehicle_severity_*.csv`
- `rush_vs_nonrush_multiyear.csv`, `dst_event_windows_multiyear.csv`
- `severity_model_metrics.csv`, `top_spatial_clusters.csv`

**Visualizations** (PNG, under `visualizations/<run>/`):
- Trend charts, heatmaps, factor comparison plots, vehicle severity charts
- Spatial scatter maps, hexbin density maps, injury hotspot overlays
- ROC curves, precision-recall curves, calibration curves
- Cluster centroid bubble plots

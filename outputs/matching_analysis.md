# Matching Analysis (Consolidated)

This file is the single consolidated matching report for subset-vs-official checks.

## Scope

Comparison target:
- Local subset: `data/NYC Accidents 2020.csv`
- Official same-window slice: `data/NYC_Collisions_Official_2020-01-01_to_2020-08-29.csv`
- Official full dataset (filtered to same window): `data/NYC_Collisions_Official_full.csv`

Window used for subset matching:
- **2020-01-01 to 2020-08-29**

Primary match key:
- `COLLISION_ID`

## Consolidated Results

| Check | Result |
|---|---|
| Local subset rows | 74,881 |
| Official same-window rows | 75,592 |
| Shared `COLLISION_ID` | 74,880 |
| IDs only in local subset | 1 |
| IDs only in official same-window slice | 712 |
| Full normalized row matches on shared IDs | 74,776 / 74,880 (99.86%) |

## What This Means

- The local 2020 file is a **valid historical subset/snapshot** of the same NYC collision source.
- Most shared rows match exactly after normalization.
- The non-overlapping rows and small field-level differences are consistent with later backfills/corrections in official data.

## Observed Small Mismatch Areas (Shared IDs)

Top mismatch columns had low counts relative to total shared rows and were concentrated in:
- Geo fields (`LATITUDE`, `LONGITUDE`, `LOCATION`)
- Some street-name fields
- A small number of injury/fatal counts
- Small number of date/time differences


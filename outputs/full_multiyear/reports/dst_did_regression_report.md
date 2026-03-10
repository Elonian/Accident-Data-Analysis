# DST DiD Regression Report

Scope: full multi-year DST DiD-style extension using compact aggregated panel.
Design:
- Separate season models: spring and fall.
- Outcome: fatal collision binary signal aggregated to 2x2 panel rates.
- Treated hours: spring 04-08, fall 17-20.
- Control hours: 10-14 (both seasons).
- Windows: 7-day pre and 7-day post around DST date, transition day excluded.

Results by season:
- Season: spring
  n_observations: 35735
  treated_pre_fatal_rate_percent: 0.138867
  treated_post_fatal_rate_percent: 0.171086
  control_pre_fatal_rate_percent: 0.109051
  control_post_fatal_rate_percent: 0.034819
  did_effect_pp: 0.106451
  coef_post_treated: 1.347951
  odds_ratio_post_treated: 3.849530
- Season: fall
  n_observations: 45958
  treated_pre_fatal_rate_percent: 0.189905
  treated_post_fatal_rate_percent: 0.194968
  control_pre_fatal_rate_percent: 0.112613
  control_post_fatal_rate_percent: 0.109804
  did_effect_pp: 0.007872
  coef_post_treated: 0.050718
  odds_ratio_post_treated: 1.052026

Output artifacts:
- tables/dst_did_panel_multiyear.csv
- tables/dst_did_regression_results_multiyear.csv
- core/dst_did_effect_multiyear.png
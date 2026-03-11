# ST-DBSCAN Temporal Clustering Report

Dataset: full_multiyear
Temporal scope: FULL dataset 2012–2026 (hour-of-day aggregated across all years)
Algorithm: ST-DBSCAN
Neighbourhood: spatial ≤ 150.0 m  AND  cyclic_hour ≤ 1.5 h
Sample: 250,000  |  Clusters: 2093  |  Noise: 32,204 (12.9%)

Top clusters:
  C  1:  96383 crashes  Brooklyn         peak 16:24  mean 14:17  peak_day Fri
  C  0:  79553 crashes  Manhattan        peak 16:25  mean 14:01  peak_day Fri
  C  7:   3234 crashes  Queens           peak 16:24  mean 13:41  peak_day Mon
  C 10:   2931 crashes  Queens           peak 13:22  mean 14:24  peak_day Tue
  C 38:   1354 crashes  Queens           peak 14:25  mean 14:43  peak_day Fri
  C 15:    993 crashes  Queens           peak 14:22  mean 14:36  peak_day Thu
  C 14:    896 crashes  Queens           peak 16:24  mean 15:01  peak_day Fri
  C 18:    752 crashes  Queens           peak 08:24  mean 13:59  peak_day Fri
  C 71:    742 crashes  Staten Island    peak 17:24  mean 15:01  peak_day Tue
  C  4:    615 crashes  Staten Island    peak 15:18  mean 15:04  peak_day Thu
  C 74:    599 crashes  Queens           peak 16:21  mean 14:27  peak_day Wed
  C 64:    510 crashes  Queens           peak 17:22  mean 15:49  peak_day Fri
  C 39:    492 crashes  Queens           peak 17:22  mean 14:12  peak_day Fri
  C177:    461 crashes  Staten Island    peak 17:17  mean 13:52  peak_day Tue
  C 31:    452 crashes  Queens           peak 17:23  mean 13:51  peak_day Fri

Outputs:
- stdbscan_spatial_scatter.png
- stdbscan_temporal_density.png
- stdbscan_cluster_profiles.png
- stdbscan_cluster_size_bar.png
# Classical CV Baseline - Experiment Results

Auto-generated from MLflow tracking data (9 runs).Regenerate with 'uv run python scripts/reporting/export_experiment_summary.py' after any new experiment. Full interactive tracking (artifacts, overlay images, per-run comparison) is available locally via 'uv run mlflow ui' after reproducing the data setup -- see docs/data_provenance.md

## exp00_circle_detection

**Parameters:**

- 'dp': 1.0
- 'image_path': C:\Users\jboze\Desktop\DSI\RetroRead\data\raw\Endava\sample_synth_datasets\ds5.0\data\v_0992_f_0000_rgba.png
- 'max_radius_fraction': 0.6
- 'min_dist_fraction': 0.5
- 'min_radius_fraction': 0.1
- 'param1': 100
- 'param2': 50

**Metrics:**

- 'center_x': 606.0000
- 'center_y': 644.0000
- 'circle_found': 1.0000
- 'radius': 167.0000

---

## exp01_batch_circle_detection

**Parameters:**

- 'accuracy_threshold_fraction': 0.2
- 'dp': 1.0
- 'max_radius_fraction': 0.6
- 'min_dist_fraction': 0.5
- 'min_radius_fraction': 0.1
- 'n_candidates': 265
- 'param1': 100
- 'param2': 50

**Metrics:**

- 'accuracy_rate': 0.9434
- 'detection_rate': 1.0000
- 'mean_center_error_px': 11.7827

---

## exp02_tighter_max_radius

**Parameters:**

- 'accuracy_threshold_fraction': 0.2
- 'changed_from_exp01': max_radius_fraction: 0.6 -> 0.35
- 'dp': 1.0
- 'max_radius_fraction': 0.35
- 'min_dist_fraction': 0.5
- 'min_radius_fraction': 0.1
- 'n_candidates': 265
- 'param1': 100
- 'param2': 50

**Metrics:**

- 'accuracy_rate': 0.9774
- 'detection_rate': 1.0000
- 'mean_center_error_px': 6.5734

---

## exp03_needle_detection_single_image

**Parameters:**

- 'canny_high': 150
- 'canny_low': 50
- 'dp': 1.0
- 'hough_threshold': 30
- 'image_path': C:\Users\jboze\Desktop\DSI\RetroRead\data\raw\Endava\sample_synth_datasets\ds5.0\data\v_0992_f_0000_rgba.png
- 'inner_radius_fraction': 0.85
- 'max_line_gap': 10
- 'max_radius_fraction': 0.35
- 'min_dist_fraction': 0.5
- 'min_line_length_fraction': 0.3
- 'min_radius_fraction': 0.1
- 'param1': 100
- 'param2': 50
- 'pivot_distance_fraction': 0.25

**Metrics:**

- 'circle_found': 1.0000
- 'needle_angle_rad': 0.2082
- 'needle_found': 1.0000
- 'needle_line_length': 126.6057

---

## exp04_batch_needle_detection

**Parameters:**

- 'accuracy_threshold_degrees': 5.0
- 'canny_high': 150
- 'canny_low': 50
- 'dp': 1.0
- 'hough_threshold': 30
- 'inner_radius_fraction': 0.85
- 'max_line_gap': 10
- 'max_radius_fraction': 0.35
- 'min_dist_fraction': 0.5
- 'min_line_length_fraction': 0.3
- 'min_radius_fraction': 0.1
- 'n_candidates': 265
- 'param1': 100
- 'param2': 50
- 'pivot_distance_fraction': 0.25

**Metrics:**

- 'accuracy_rate': 0.9094
- 'mean_angle_error_deg': 4.4401
- 'mean_tip_error_px': 15.1937
- 'needle_detection_rate': 0.9585

---

## exp05_reading_conversion_single_image

**Parameters:**

- 'image_path': data/v_0992_f_0000_rgba.png

**Metrics:**

- 'math_check_error': 0.2987
- 'math_check_predicted': 8.8073
- 'pipeline_error': 0.3399
- 'pipeline_predicted': 8.7660
- 'true_value': 9.1059

---

## exp06_batch_reading_conversion

**Parameters:**

- 'canny_high': 150
- 'canny_low': 50
- 'dp': 1.0
- 'hough_threshold': 30
- 'inner_radius_fraction': 0.85
- 'max_line_gap': 10
- 'max_radius_fraction': 0.35
- 'min_dist_fraction': 0.5
- 'min_line_length_fraction': 0.3
- 'min_radius_fraction': 0.1
- 'n_candidates': 265
- 'param1': 100
- 'param2': 50
- 'pivot_distance_fraction': 0.25
- 'tolerance_pct': 5.0

**Metrics:**

- 'accuracy_rate_within_tolerance': 0.8415
- 'mean_pct_error': 4.2619
- 'mean_raw_error': 0.3425
- 'median_pct_error': 2.8699
- 'needle_detection_rate': 0.9585

---

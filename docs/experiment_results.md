# All Experiments — MLflow Export

Auto-generated from MLflow tracking data. Regenerate with `uv run python scripts/reporting/export_experiment_summary.py` after any new experiment. Full interactive tracking (artifacts, overlay images, per-run comparison) is available locally via `uv run mlflow ui` after reproducing the data setup -- see docs/data_provenance.md

## retroread_classical_baseline (18 runs)

### exp05_reading_conversion_single_image

**Parameters:**

- `image_path`: data/v_0992_f_0000_rgba.png

**Metrics:**

- `math_check_error`: 0.2987
- `math_check_predicted`: 8.8073
- `pipeline_error`: 0.3399
- `pipeline_predicted`: 8.7660
- `true_value`: 9.1059

---

### exp06_batch_reading_conversion

**Parameters:**

- `canny_high`: 150
- `canny_low`: 50
- `dp`: 1.0
- `hough_threshold`: 30
- `inner_radius_fraction`: 0.85
- `max_line_gap`: 10
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_line_length_fraction`: 0.3
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50
- `pivot_distance_fraction`: 0.25
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate_within_tolerance`: 0.8415
- `mean_pct_error`: 4.2619
- `mean_raw_error`: 0.3425
- `median_pct_error`: 2.8720
- `needle_detection_rate`: 0.9585

---

### exp00_circle_detection

**Parameters:**

- `dp`: 1.0
- `image_path`: C:\Users\jboze\Desktop\DSI\RetroRead\data\raw\Endava\sample_synth_datasets\ds5.0\data\v_0992_f_0000_rgba.png
- `max_radius_fraction`: 0.6
- `min_dist_fraction`: 0.5
- `min_radius_fraction`: 0.1
- `param1`: 100
- `param2`: 50

**Metrics:**

- `center_x`: 606.0000
- `center_y`: 644.0000
- `circle_found`: 1.0000
- `radius`: 167.0000

---

### exp01_batch_circle_detection

**Parameters:**

- `accuracy_threshold_fraction`: 0.2
- `dp`: 1.0
- `max_radius_fraction`: 0.6
- `min_dist_fraction`: 0.5
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50

**Metrics:**

- `accuracy_rate`: 0.9434
- `detection_rate`: 1.0000
- `mean_center_error_px`: 11.7827

---

### exp02_tighter_max_radius

**Parameters:**

- `accuracy_threshold_fraction`: 0.2
- `changed_from_exp01`: max_radius_fraction: 0.6 -> 0.35
- `dp`: 1.0
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50

**Metrics:**

- `accuracy_rate`: 0.9774
- `detection_rate`: 1.0000
- `mean_center_error_px`: 6.5734

---

### exp03_needle_detection_single_image

**Parameters:**

- `canny_high`: 150
- `canny_low`: 50
- `dp`: 1.0
- `hough_threshold`: 30
- `image_path`: C:\Users\jboze\Desktop\DSI\RetroRead\data\raw\Endava\sample_synth_datasets\ds5.0\data\v_0992_f_0000_rgba.png
- `inner_radius_fraction`: 0.85
- `max_line_gap`: 10
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_line_length_fraction`: 0.3
- `min_radius_fraction`: 0.1
- `param1`: 100
- `param2`: 50
- `pivot_distance_fraction`: 0.25

**Metrics:**

- `circle_found`: 1.0000
- `needle_angle_rad`: 0.2082
- `needle_found`: 1.0000
- `needle_line_length`: 126.6057

---

### exp04_batch_needle_detection

**Parameters:**

- `accuracy_threshold_degrees`: 5.0
- `canny_high`: 150
- `canny_low`: 50
- `dp`: 1.0
- `hough_threshold`: 30
- `inner_radius_fraction`: 0.85
- `max_line_gap`: 10
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_line_length_fraction`: 0.3
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50
- `pivot_distance_fraction`: 0.25

**Metrics:**

- `accuracy_rate`: 0.9094
- `mean_angle_error_deg`: 4.4401
- `mean_tip_error_px`: 15.1937
- `needle_detection_rate`: 0.9585

---

### exp05_reading_conversion_single_image

**Parameters:**

- `image_path`: data/v_0992_f_0000_rgba.png

**Metrics:**

- `math_check_error`: 0.2987
- `math_check_predicted`: 8.8073
- `pipeline_error`: 0.3399
- `pipeline_predicted`: 8.7660
- `true_value`: 9.1059

---

### exp06_batch_reading_conversion

**Parameters:**

- `canny_high`: 150
- `canny_low`: 50
- `dp`: 1.0
- `hough_threshold`: 30
- `inner_radius_fraction`: 0.85
- `max_line_gap`: 10
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_line_length_fraction`: 0.3
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50
- `pivot_distance_fraction`: 0.25
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate_within_tolerance`: 0.8415
- `mean_pct_error`: 4.2619
- `mean_raw_error`: 0.3425
- `median_pct_error`: 2.8699
- `needle_detection_rate`: 0.9585

---

### exp04_batch_needle_detection

**Parameters:**

- `accuracy_threshold_degrees`: 5.0
- `canny_high`: 150
- `canny_low`: 50
- `dp`: 1.0
- `hough_threshold`: 30
- `inner_radius_fraction`: 0.85
- `max_line_gap`: 10
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_line_length_fraction`: 0.3
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50
- `pivot_distance_fraction`: 0.25

**Metrics:**

- `accuracy_rate`: 0.9094
- `mean_angle_error_deg`: 4.4401
- `mean_tip_error_px`: 15.1937
- `needle_detection_rate`: 0.9585

---

### exp06_batch_reading_conversion

**Parameters:**

- `canny_high`: 150
- `canny_low`: 50
- `dp`: 1.0
- `hough_threshold`: 30
- `inner_radius_fraction`: 0.85
- `max_line_gap`: 10
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_line_length_fraction`: 0.3
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50
- `pivot_distance_fraction`: 0.25
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate_within_tolerance`: 0.8415
- `mean_pct_error`: 4.2619
- `mean_raw_error`: 0.3425
- `median_pct_error`: 2.8699
- `needle_detection_rate`: 0.9585

---

### exp06_batch_reading_conversion

**Parameters:**

- `canny_high`: 150
- `canny_low`: 50
- `dp`: 1.0
- `hough_threshold`: 30
- `inner_radius_fraction`: 0.85
- `max_line_gap`: 10
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_line_length_fraction`: 0.3
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50
- `pivot_distance_fraction`: 0.25
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate_within_tolerance`: 0.8415
- `mean_pct_error`: 4.2619
- `mean_raw_error`: 0.3425
- `median_pct_error`: 2.8699
- `needle_detection_rate`: 0.9585

---

### exp06_batch_reading_conversion

**Parameters:**

- `canny_high`: 150
- `canny_low`: 50
- `dp`: 1.0
- `hough_threshold`: 30
- `inner_radius_fraction`: 0.85
- `max_line_gap`: 10
- `max_radius_fraction`: 0.35
- `min_dist_fraction`: 0.5
- `min_line_length_fraction`: 0.3
- `min_radius_fraction`: 0.1
- `n_candidates`: 265
- `param1`: 100
- `param2`: 50
- `pivot_distance_fraction`: 0.25
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate_within_tolerance`: 0.8415
- `mean_pct_error`: 4.2619
- `mean_raw_error`: 0.3425
- `median_pct_error`: 2.8699
- `needle_detection_rate`: 0.9585

---

### exp06b_classical_baseline_val_split

**Parameters:**

- `n_candidates`: 61

**Metrics:**

- `accuracy_rate`: 0.8033
- `mean_pct_error`: 4.8241
- `needle_detection_rate`: 0.9836

---

### exp19_confidence_abstention

**Parameters:**


**Metrics:**

- `accuracy_t0.0`: 0.8167
- `accuracy_t0.3`: 0.8167
- `accuracy_t0.4`: 0.8167
- `accuracy_t0.5`: 0.8167
- `accuracy_t0.6`: 0.8167
- `accuracy_t0.7`: 0.8167
- `accuracy_t0.8`: 0.8167
- `coverage_t0.0`: 0.9836
- `coverage_t0.3`: 0.9836
- `coverage_t0.4`: 0.9836
- `coverage_t0.5`: 0.9836
- `coverage_t0.6`: 0.9836
- `coverage_t0.7`: 0.9836
- `coverage_t0.8`: 0.9836
- `mean_error_t0.0`: 4.8241
- `mean_error_t0.3`: 4.8241
- `mean_error_t0.4`: 4.8241
- `mean_error_t0.5`: 4.8241
- `mean_error_t0.6`: 4.8241
- `mean_error_t0.7`: 4.8241
- `mean_error_t0.8`: 4.8241

---

### exp19_confidence_abstention

**Parameters:**


**Metrics:**

- `accuracy_t0.0`: 0.8167
- `accuracy_t0.3`: 0.8167
- `accuracy_t0.4`: 0.8167
- `accuracy_t0.5`: 0.8167
- `accuracy_t0.6`: 0.8167
- `accuracy_t0.7`: 0.8167
- `accuracy_t0.8`: 0.8305
- `coverage_t0.0`: 0.9836
- `coverage_t0.3`: 0.9836
- `coverage_t0.4`: 0.9836
- `coverage_t0.5`: 0.9836
- `coverage_t0.6`: 0.9836
- `coverage_t0.7`: 0.9836
- `coverage_t0.8`: 0.9672
- `mean_error_t0.0`: 4.8241
- `mean_error_t0.3`: 4.8241
- `mean_error_t0.4`: 4.8241
- `mean_error_t0.5`: 4.8241
- `mean_error_t0.6`: 4.8241
- `mean_error_t0.7`: 4.8241
- `mean_error_t0.8`: 3.7676

---

### exp19_confidence_abstention

**Parameters:**


**Metrics:**

- `accuracy_t0.0`: 0.8167
- `accuracy_t0.5`: 0.8167
- `accuracy_t0.7`: 0.8167
- `accuracy_t0.8`: 0.8305
- `accuracy_t0.85`: 0.8448
- `accuracy_t0.9`: 0.8448
- `accuracy_t0.95`: 0.8448
- `accuracy_t0.99`: 0.8448
- `coverage_t0.0`: 0.9836
- `coverage_t0.5`: 0.9836
- `coverage_t0.7`: 0.9836
- `coverage_t0.8`: 0.9672
- `coverage_t0.85`: 0.9508
- `coverage_t0.9`: 0.9508
- `coverage_t0.95`: 0.9508
- `coverage_t0.99`: 0.9508
- `mean_error_t0.0`: 4.8241
- `mean_error_t0.5`: 4.8241
- `mean_error_t0.7`: 4.8241
- `mean_error_t0.8`: 3.7676
- `mean_error_t0.85`: 3.4067
- `mean_error_t0.9`: 3.4067
- `mean_error_t0.95`: 3.4067
- `mean_error_t0.99`: 3.4067

---

### exp19_confidence_abstention

**Parameters:**


**Metrics:**

- `accuracy_t0.0`: 0.8167
- `accuracy_t0.5`: 0.8167
- `accuracy_t0.7`: 0.8167
- `accuracy_t0.8`: 0.8305
- `accuracy_t0.85`: 0.8448
- `accuracy_t0.9`: 0.8448
- `accuracy_t0.95`: 0.8448
- `accuracy_t0.99`: 0.8448
- `coverage_t0.0`: 0.9836
- `coverage_t0.5`: 0.9836
- `coverage_t0.7`: 0.9836
- `coverage_t0.8`: 0.9672
- `coverage_t0.85`: 0.9508
- `coverage_t0.9`: 0.9508
- `coverage_t0.95`: 0.9508
- `coverage_t0.99`: 0.9508
- `mean_error_t0.0`: 4.8241
- `mean_error_t0.5`: 4.8241
- `mean_error_t0.7`: 4.8241
- `mean_error_t0.8`: 3.7676
- `mean_error_t0.85`: 3.4067
- `mean_error_t0.9`: 3.4067
- `mean_error_t0.95`: 3.4067
- `mean_error_t0.99`: 3.4067

---

## retroread_deep_learning (48 runs)

### exp08_training_sanity_check

**Parameters:**

- `batch_size`: 6
- `learning_rate`: 0.001
- `n_epochs`: 5
- `n_subset`: 30

**Metrics:**


---

### exp08_training_sanity_check

**Parameters:**

- `batch_size`: 6
- `learning_rate`: 0.001
- `n_epochs`: 5
- `n_subset`: 30

**Metrics:**

- `train_loss`: 1.0030

---

### exp08_training_sanity_check

**Parameters:**

- `batch_size`: 6
- `learning_rate`: 0.001
- `n_epochs`: 5
- `n_subset`: 30

**Metrics:**

- `train_loss`: 1.0019

---

### exp08_training_sanity_check

**Parameters:**

- `batch_size`: 6
- `learning_rate`: 0.001
- `n_epochs`: 5
- `n_subset`: 30

**Metrics:**

- `train_loss`: 0.0138

---

### exp08_training_sanity_check

**Parameters:**

- `batch_size`: 6
- `learning_rate`: 0.001
- `n_epochs`: 5
- `n_subset`: 30

**Metrics:**

- `train_loss`: 0.0145

---

### exp09_full_training

**Parameters:**

- `batch_size`: 16
- `learning_rate`: 0.001
- `n_epochs`: 10
- `n_train`: 800
- `n_val`: 200

**Metrics:**

- `best_val_loss`: 0.0160
- `train_loss`: 0.0114
- `val_loss`: 0.0178

---

### exp10_dl_evaluation

**Parameters:**

- `checkpoint`: best_model.pt
- `n_val_candidates`: 200
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate`: 0.0100
- `mean_pct_error`: 42.3390

---

### exp09_full_training

**Parameters:**

- `batch_size`: 16
- `learning_rate`: 0.001
- `n_epochs`: 10
- `n_train`: 800
- `n_val`: 200

**Metrics:**

- `best_val_loss`: 0.0050
- `train_loss`: 0.0008
- `val_loss`: 0.0077

---

### exp10_dl_evaluation

**Parameters:**

- `checkpoint`: best_model.pt
- `n_val_candidates`: 200
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate`: 0.1250
- `mean_pct_error`: 31.0932

---

### exp09_full_training

**Parameters:**

- `batch_size`: 16
- `learning_rate`: 0.001
- `n_epochs`: 10
- `n_train`: 800
- `n_val`: 200

**Metrics:**

- `best_val_loss`: 0.0063
- `train_loss`: 0.0008
- `val_loss`: 0.0077

---

### exp10_dl_evaluation

**Parameters:**

- `checkpoint`: best_model.pt
- `n_val_candidates`: 200
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate`: 0.1200
- `mean_pct_error`: 30.8930

---

### exp09_full_training

**Parameters:**

- `batch_size`: 16
- `learning_rate`: 0.0005
- `n_epochs`: 10
- `n_train`: 800
- `n_val`: 200

**Metrics:**

- `best_val_loss`: 0.0058
- `train_loss`: 0.0003
- `val_loss`: 0.0071

---

### exp10_dl_evaluation

**Parameters:**

- `checkpoint`: best_model.pt
- `n_val_candidates`: 200
- `tolerance_pct`: 5.0

**Metrics:**

- `accuracy_rate`: 0.0900
- `mean_pct_error`: 38.8909

---

### exp11_heatmap_training

**Parameters:**

- `architecture`: heatmap_upsample
- `batch_size`: 16
- `learning_rate`: 0.0005
- `n_epochs`: 10
- `n_train`: 800
- `n_val`: 200

**Metrics:**

- `best_val_loss`: 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0000
- `train_loss`: nan
- `val_loss`: nan

---

### exp11_heatmap_training

**Parameters:**

- `architecture`: heatmap_upsample
- `batch_size`: 16
- `learning_rate`: 0.0005
- `n_epochs`: 10
- `n_train`: 800
- `n_val`: 200

**Metrics:**

- `best_val_loss`: 0.0022
- `train_loss`: 0.0016
- `val_loss`: 0.0022

---

### exp12_heatmap_evaluation

**Parameters:**

- `checkpoint`: best_heatmap_model.pt

**Metrics:**

- `accuracy_rate`: 0.2950
- `mean_pct_error`: 15.9964

---

### exp11_heatmap_training

**Parameters:**

- `architecture`: heatmap_upsample
- `batch_size`: 16
- `learning_rate`: 0.0005
- `n_epochs`: 20
- `n_train`: 800
- `n_val`: 200

**Metrics:**

- `best_val_loss`: 0.0015
- `train_loss`: 0.0003
- `val_loss`: 0.0016

---

### exp12_heatmap_evaluation

**Parameters:**

- `checkpoint`: best_heatmap_model.pt

**Metrics:**

- `accuracy_rate`: 0.4650
- `mean_pct_error`: 12.0871

---

### exp13_crop_heatmap_training

**Parameters:**

- `architecture`: crop_heatmap_skip_softargmax
- `batch_size`: 16
- `coord_loss_weight`: 5.0
- `learning_rate`: 0.0005
- `n_epochs`: 25
- `n_train`: 800
- `n_val`: 200

**Metrics:**


---

### exp13_crop_heatmap_training

**Parameters:**

- `architecture`: crop_heatmap_skip_softargmax
- `batch_size`: 16
- `coord_loss_weight`: 5.0
- `learning_rate`: 0.0005
- `n_epochs`: 25
- `n_train`: 800
- `n_val`: 200

**Metrics:**


---

### exp13_crop_heatmap_training

**Parameters:**

- `architecture`: crop_heatmap_skip_softargmax
- `batch_size`: 16
- `coord_loss_weight`: 5.0
- `learning_rate`: 0.0005
- `n_epochs`: 25
- `n_train`: 800
- `n_val`: 200

**Metrics:**


---

### exp13_crop_heatmap_training

**Parameters:**

- `architecture`: crop_heatmap_skip_softargmax
- `batch_size`: 16
- `coord_loss_weight`: 5.0
- `learning_rate`: 0.0005
- `n_epochs`: 25
- `n_train`: 800
- `n_val`: 200

**Metrics:**

- `best_val_loss`: 0.2591
- `train_loss`: 0.2275
- `val_coord_loss`: 0.0454
- `val_loss`: 0.2636

---

### exp14_crop_heatmap_evaluation

**Parameters:**

- `checkpoint`: best_crop_heatmap_model.pt

**Metrics:**

- `accuracy_rate`: 0.5100
- `mean_pct_error`: 8.9146

---

### exp14_crop_heatmap_evaluation

**Parameters:**

- `checkpoint`: best_crop_heatmap_model.pt

**Metrics:**

- `accuracy_rate`: 0.5410
- `mean_pct_error`: 7.2753

---

### exp15_differential_lr_3blocks

**Parameters:**

- `existing_block_lr`: 5e-05
- `head_lr`: 0.0005
- `n_epochs`: 25
- `n_unfrozen_blocks`: 3
- `new_blocks_lr`: 2e-05
- `resumed_from`: best_crop_heatmap_model.pt

**Metrics:**

- `best_val_loss`: 0.2499
- `train_loss`: 0.2184
- `val_loss`: 0.2499

---

### exp16_differential_lr_evaluation

**Parameters:**

- `checkpoint`: best_differential_lr_model.pt

**Metrics:**

- `accuracy_rate`: 0.4754
- `mean_pct_error`: 8.8828

---

### exp17_reading_aware_loss

**Parameters:**

- `coord_loss_weight`: 5.0
- `learning_rate`: 0.0005
- `n_epochs`: 25
- `reading_loss_weight`: 2.0
- `resumed_from`: best_crop_heatmap_model.pt

**Metrics:**

- `best_val_loss`: 0.5771
- `train_loss`: 0.4933
- `val_loss`: 0.5771
- `val_reading_frac_error`: 0.0639

---

### exp14_crop_heatmap_evaluation

**Parameters:**

- `checkpoint`: best_reading_aware_model.pt

**Metrics:**

- `accuracy_rate`: 0.1475
- `mean_pct_error`: 25.6308

---

### exp20_crop_jitter_training

**Parameters:**

- `coord_loss_weight`: 5.0
- `learning_rate`: 0.0005
- `n_epochs`: 25
- `position_jitter`: 0.1
- `resumed_from`: best_crop_heatmap_model.pt
- `scale_jitter`: 0.15
- `validation_crop_source`: real_classical_detector

**Metrics:**

- `best_val_loss`: 0.1557
- `train_loss`: 0.1246
- `val_loss_detector_crop`: 0.1638

---

### exp21_crop_jitter_evaluation

**Parameters:**

- `checkpoint`: best_crop_jitter_model.pt

**Metrics:**

- `accuracy_rate`: 0.6721
- `mean_pct_error`: 5.5227

---

### exp22_needle_segmentation_training

**Parameters:**

- `architecture`: needle_segmentation_plus_center_heatmap
- `learning_rate`: 0.0005
- `n_epochs`: 25

**Metrics:**

- `best_val_loss`: 0.0302
- `train_loss`: 0.0304
- `val_loss`: 0.0302

---

### exp23_needle_segmentation_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.6721
- `mean_pct_error`: 6.8012

---

### exp24_multitask_training

**Parameters:**

- `architecture`: multitask_mask_center_landmarks
- `n_epochs`: 25

**Metrics:**


---

### exp24_multitask_training

**Parameters:**

- `architecture`: multitask_mask_center_landmarks
- `n_epochs`: 25

**Metrics:**

- `best_val_loss`: 0.0279
- `train_loss`: 0.0284
- `val_loss`: 0.0279

---

### exp25_multitask_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.0820
- `mean_pct_error`: 71.0641

---

### exp25_multitask_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.0820
- `mean_pct_error`: 71.0641

---

### exp26_angle_regression_training

**Parameters:**

- `architecture`: sincos_angle_regression
- `n_epochs`: 25

**Metrics:**

- `best_val_loss`: 0.0756
- `train_loss`: 0.0758
- `val_loss`: 0.1909

---

### exp27_angle_regression_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.3279
- `mean_pct_error`: 13.5345

---

### exp28_direct_regression_training

**Parameters:**

- `architecture`: direct_normalized_position_regression
- `n_epochs`: 25

**Metrics:**

- `best_val_loss`: 0.0221
- `train_loss`: 0.0181
- `val_loss`: 0.0507

---

### exp29_direct_regression_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.3279
- `mean_pct_error`: 17.2324

---

### exp31_yolo_pose_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.7213
- `mean_pct_error`: 5.1415

---

### exp26_angle_regression_training

**Parameters:**

- `architecture`: sincos_angle_regression
- `n_epochs`: 25

**Metrics:**

- `best_val_loss`: 0.0741
- `train_loss`: 0.0666
- `val_loss`: 0.2792

---

### exp27_angle_regression_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.4426
- `mean_pct_error`: 16.8839

---

### exp28_direct_regression_training

**Parameters:**

- `architecture`: direct_normalized_position_regression
- `n_epochs`: 25

**Metrics:**

- `best_val_loss`: 0.1490
- `train_loss`: 0.0629
- `val_loss`: 0.2064

---

### exp29_direct_regression_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.0000
- `mean_pct_error`: 37.3536

---

### exp33_p2_yolo_pose_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.2459
- `mean_pct_error`: 13.2933

---

### exp34_tip_refiner_training

**Parameters:**

- `jitter_range`: 32-56px
- `n_epochs`: 25
- `roi_size`: 128

**Metrics:**

- `best_val_loss`: 0.0015
- `train_loss`: 0.0013
- `val_loss`: 0.0015

---

### exp35_tip_refiner_evaluation

**Parameters:**


**Metrics:**

- `accuracy_rate`: 0.7705
- `mean_pct_error`: 4.3070

---

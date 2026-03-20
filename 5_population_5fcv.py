#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:41:13 2026

@author: temuuleu
"""



PLOT = False
SAVE_METRICS = True
MAX_RETRIES = 30
RETRY_DELAY = 2


seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
if DEBUG:
    seeds = [1]

N_FOLDS = 5
SCANNER_NAMES = ['Philips', 'Tim Trio', 'Prisma_fit']
thresholds_list = [0.85, 0.90, 'locate']

train_image_directory_path = '/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/DATASET_SC/PROJECT_NULL_07_04_2025/1_Project_all_code_all_data/Phase_1/LOCATE_SET/locate_train'
LOCATE_PATH                = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Projects/BIANCA/LOCATE/LOCATE-BIANCA"
feature_select = [1, 1, 1, 1]

bianca_pool       = "Phase_1/LOCATE_SET/bianca_pool_wihtouth_ge.xlsx"
bianca_pool_df    = pd.read_excel(bianca_pool)
CV_SET_base_dir = "Phase_1/5FCV_SET"
all_metrics_result_dir = os.path.join(CV_SET_base_dir, "bianca_result_dir")
os.makedirs(all_metrics_result_dir, exist_ok=True)
png_images = os.path.join(CV_SET_base_dir, "png_images")
os.makedirs(png_images, exist_ok=True)




# =============================================================
# COMBINE ALL SEED RESULTS
# =============================================================
all_seed_files = glob.glob(os.path.join(all_metrics_result_dir, "bianca_metrics_seed_*.xlsx"))
metrics_output_path = os.path.join(CV_SET_base_dir, "bianca_evaluation_metrics.xlsx")

#here i want to check if all subject and methods etc are run

if not DEBUG and all_seed_files:
    combined_df = pd.concat([pd.read_excel(f) for f in all_seed_files], ignore_index=True)

    # --- Completeness Check ---
    expected_seeds = set(seeds)
    expected_folds = set(range(N_FOLDS))
    expected_thresholds = set(['85', '90', 'locate'])
    expected_train_conds = set([c['name'] for c in train_conditions])
    expected_test_conds = set([c['name'] for c in test_conditions])

    found_seeds = set(combined_df['seed'].unique())
    missing_seeds = expected_seeds - found_seeds
    if missing_seeds:
        print(f"WARNING: Missing seeds: {sorted(missing_seeds)}")

    for s in sorted(found_seeds):
        seed_df = combined_df[combined_df['seed'] == s]
        found_folds = set(seed_df['fold'].unique())
        
        
        missing_folds = expected_folds - found_folds
        if missing_folds:
            print(f"  WARNING: Seed {s} missing folds: {sorted(missing_folds)}")

        for f in sorted(found_folds):
            fold_df = seed_df[seed_df['fold'] == f]
            combos = fold_df.groupby(['train_condition', 'test_condition', 'threshold'])['subject'].nunique()
            n_test_expected = len(fold_df['subject'].unique())

            for tc in expected_train_conds:
                for te in expected_test_conds:
                    for th in expected_thresholds:
                        try:
                            n = combos.loc[(tc, te, th)]
                        except KeyError:
                            n = 0
                        if n < n_test_expected:
                            print(f"  WARNING: Seed {s} Fold {f} train_{tc}__test_{te} thresh={th}: "
                                  f"{n}/{n_test_expected} subjects")

    total_expected = len(seeds) * N_FOLDS * len(expected_train_conds) * len(expected_test_conds) * len(expected_thresholds)
    # expected per combo = avg test subjects per fold
    print(f"\nCompleteness: {len(combined_df)} rows total")
    print(f"Seeds found: {sorted(found_seeds)} / expected: {sorted(expected_seeds)}")
    print(f"Seed files: {len(all_seed_files)} / {len(seeds)}")
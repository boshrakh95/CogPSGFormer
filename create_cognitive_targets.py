# Create cognitive targets for STAGES (regression and classification).
# Create composites scores for different cognitive aspects of each test

import numpy as np
import pandas as pd
import re
import glob2
from scipy.stats import zscore
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def n_back_scores(dir_targets, subj_to_ret, method):

    if method == 'subsets_separately':
        nback_working_memory_capacity, nback_speed, nback_accuracy, nback_impulsivity, \
            subj_ids_all = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        for tar in range(len(dir_targets)):
            # tar = tar
            targets = pd.read_excel(dir_targets[tar])

            # List of columns to check for missing values
            columns_to_check = ['SLNB12_60.SLNB2_TP', 'SLNB12_60.SLNB2_FP', 'SLNB12_60.SLNB2_RTC',
                                'SLNB12_60.SLNB2_MCR', 'SLNB12_60.SLNB2_MRTC', 'SLNB12_60.SLNB2_FP1',
                                'SLNB12_60.SLNB2_RTC1', 'SLNB12_60.SLNB2_FP2', 'SLNB12_60.SLNB2_RTC2']
            # Find rows with non-missing values for all specified columns
            # Assuming 'subject_id' is the column name for subject IDs in your DataFrame
            targets1 = targets.dropna(subset=columns_to_check)

            # Get the subject IDs and index of rows that have all required values
            print("Number of all subjects in ", dir_targets[tar], "is: ", len(targets))
            subj_ids = targets1['test_sessions.subid'].to_numpy()
            print("Number of subjects with test scores in ", dir_targets[tar], "is: ", len(subj_ids))
            # indices_with_all_values = targets1.index.tolist()

            # Extract the desired scores and z-score them
            total_true_positives = zscore(targets1['SLNB12_60.SLNB2_TP']).to_numpy()
            total_false_positives = zscore(targets1['SLNB12_60.SLNB2_FP']).to_numpy()
            median_rt_correct = zscore(targets1['SLNB12_60.SLNB2_RTC']).to_numpy()
            true_positive_1_2_back = zscore(targets1['SLNB12_60.SLNB2_MCR']).to_numpy()
            median_rt_correct_1_2_back = zscore(targets1['SLNB12_60.SLNB2_MRTC']).to_numpy()
            # true_positive_1_back = zscore(targets1['SLNB12_60.SLNB2_TP1']).to_numpy()
            false_positive_1_back = zscore(targets1['SLNB12_60.SLNB2_FP1']).to_numpy()
            median_rt_tp_1_back = zscore(targets1['SLNB12_60.SLNB2_RTC1']).to_numpy()
            # median_rt_1_back = zscore(targets1['SLNB12_60.SLNB2_RT1']).to_numpy()
            # true_positive_2_back = zscore(targets1['SLNB12_60.SLNB2_TP2']).to_numpy()
            false_positive_2_back = zscore(targets1['SLNB12_60.SLNB2_FP2']).to_numpy()
            median_rt_tp_2_back = zscore(targets1['SLNB12_60.SLNB2_RTC2']).to_numpy()
            # median_rt_2_back = zscore(targets1['SLNB12_60.SLNB2_RT2']).to_numpy()

            # Compute the composite scores for each cognitive aspect by summing the z-scored values
            # (+ for good performance, - for bad)
            working_memory_capacity = total_true_positives - total_false_positives - median_rt_correct
            speed = -median_rt_correct - median_rt_tp_1_back - median_rt_tp_2_back - median_rt_correct_1_2_back
            accuracy = true_positive_1_2_back - false_positive_1_back - false_positive_2_back
            impulsivity = total_false_positives

            # Concatenate the values from all the subsets
            nback_working_memory_capacity = np.concatenate((nback_working_memory_capacity, working_memory_capacity))
            nback_speed = np.concatenate((nback_speed, speed))
            nback_accuracy = np.concatenate((nback_accuracy, accuracy))
            nback_impulsivity = np.concatenate((nback_impulsivity, impulsivity))
            subj_ids_all = np.concatenate((subj_ids_all, subj_ids))

    elif method == 'all_subsets_together':

        # Initialize the arrays for all scores arrays to store the values
        total_true_positives_all, total_false_positives_all, median_rt_correct_all, true_positive_1_2_back_all, \
            median_rt_correct_1_2_back_all, false_positive_1_back_all, median_rt_tp_1_back_all, false_positive_2_back_all, \
            median_rt_tp_2_back_all, subj_ids_all = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        for tar in range(len(dir_targets)):

            targets = pd.read_excel(dir_targets[tar])

            # List of columns to check for missing values
            columns_to_check = ['SLNB12_60.SLNB2_TP', 'SLNB12_60.SLNB2_FP', 'SLNB12_60.SLNB2_RTC',
                                'SLNB12_60.SLNB2_MCR', 'SLNB12_60.SLNB2_MRTC', 'SLNB12_60.SLNB2_FP1',
                                'SLNB12_60.SLNB2_RTC1', 'SLNB12_60.SLNB2_FP2', 'SLNB12_60.SLNB2_RTC2']
            # Find rows with non-missing values for all specified columns
            # Assuming 'subject_id' is the column name for subject IDs in your DataFrame
            targets1 = targets.dropna(subset=columns_to_check)

            # Get the subject IDs and index of rows that have all required values
            print("Number of all subjects in ", dir_targets[tar], "is: ", len(targets))
            subj_ids = targets1['test_sessions.subid'].to_numpy()
            print("Number of subjects with test scores in ", dir_targets[tar], "is: ", len(subj_ids))
            # indices_with_all_values = targets1.index.tolist()

            # Extract the desired scores
            total_true_positives = targets1['SLNB12_60.SLNB2_TP'].to_numpy()
            total_false_positives = targets1['SLNB12_60.SLNB2_FP'].to_numpy()
            median_rt_correct = targets1['SLNB12_60.SLNB2_RTC'].to_numpy()
            true_positive_1_2_back = targets1['SLNB12_60.SLNB2_MCR'].to_numpy()
            median_rt_correct_1_2_back = targets1['SLNB12_60.SLNB2_MRTC'].to_numpy()
            # true_positive_1_back = targets1['SLNB12_60.SLNB2_TP1'].to_numpy()
            false_positive_1_back = targets1['SLNB12_60.SLNB2_FP1'].to_numpy()
            median_rt_tp_1_back = targets1['SLNB12_60.SLNB2_RTC1'].to_numpy()
            # median_rt_1_back = targets1['SLNB12_60.SLNB2_RT1'].to_numpy()
            # true_positive_2_back = targets1['SLNB12_60.SLNB2_TP2'].to_numpy()
            false_positive_2_back = targets1['SLNB12_60.SLNB2_FP2'].to_numpy()
            median_rt_tp_2_back = targets1['SLNB12_60.SLNB2_RTC2'].to_numpy()
            # median_rt_2_back = targets1['SLNB12_60.SLNB2_RT2'].to_numpy()

            # Concatenate the values from all the subsets
            total_true_positives_all = np.concatenate((total_true_positives_all, total_true_positives))
            total_false_positives_all = np.concatenate((total_false_positives_all, total_false_positives))
            median_rt_correct_all = np.concatenate((median_rt_correct_all, median_rt_correct))
            true_positive_1_2_back_all = np.concatenate((true_positive_1_2_back_all, true_positive_1_2_back))
            median_rt_correct_1_2_back_all = np.concatenate((median_rt_correct_1_2_back_all, median_rt_correct_1_2_back))
            false_positive_1_back_all = np.concatenate((false_positive_1_back_all, false_positive_1_back))
            median_rt_tp_1_back_all = np.concatenate((median_rt_tp_1_back_all, median_rt_tp_1_back))
            false_positive_2_back_all = np.concatenate((false_positive_2_back_all, false_positive_2_back))
            median_rt_tp_2_back_all = np.concatenate((median_rt_tp_2_back_all, median_rt_tp_2_back))
            subj_ids_all = np.concatenate((subj_ids_all, subj_ids))

        # Compute zscore of each score and then the composite scores for each cognitive aspect by
        # summing the z-scored values (+ for good performance, - for bad)
        nback_working_memory_capacity = zscore(total_true_positives_all) - zscore(total_false_positives_all) - zscore(median_rt_correct_all)
        nback_speed = -zscore(median_rt_correct_all) - zscore(median_rt_tp_1_back_all) - zscore(median_rt_tp_2_back_all) - zscore(median_rt_correct_1_2_back_all)
        nback_accuracy = zscore(true_positive_1_2_back_all) - zscore(false_positive_1_back_all) - zscore(false_positive_2_back_all)
        nback_impulsivity = zscore(total_false_positives_all)

    names_ret = []
    for num_subj in range(len(subj_to_ret)):
        m = re.search('.*/([^/]+)$', subj_to_ret[num_subj])
        if m:
            name = m.group(1)
        # name = name.replace('/', '')
        names_ret.append(name)
    names_ret = np.array(names_ret)  # 935 (subjs retained for power analysis)

    sub_id_retain = [index for index, subject in enumerate(subj_ids_all) if subject in names_ret]  # 735 (subjs retained for nback analysis)
    names_ret_nback = [subject for subject in names_ret if subject in subj_ids_all]  # // //
    # missing_subjects = [subject for subject in names_ret if subject not in subj_ids_all]  # 200 (subjs from power analysis without nback test scores)

    # Retain only the subjects with n-back scores
    nback_working_memory_capacity = nback_working_memory_capacity[sub_id_retain]
    nback_speed = nback_speed[sub_id_retain]
    nback_accuracy = nback_accuracy[sub_id_retain]
    nback_impulsivity = nback_impulsivity[sub_id_retain]

    # Save regression values (all 4 in one array with column names and subject IDs)
    nback_all = pd.DataFrame({
        'subject_id': names_ret_nback,
        'working_memory_capacity': nback_working_memory_capacity,
        'speed': nback_speed,
        'accuracy': nback_accuracy,
        'impulsivity': nback_impulsivity
    })
    # nback_all.to_csv(path_save + "/nback_scores_regression_for_yasa_c3_eeg_rel_power_analysis.csv", index=False)

    # Convert regression to binary classification for each cognitive aspect using the median as threshold
    thresh_wmc = np.median(nback_working_memory_capacity)
    print(np.sum(nback_working_memory_capacity >= np.median(nback_working_memory_capacity)), "subj with high wmc")
    print(np.sum(nback_working_memory_capacity < np.median(nback_working_memory_capacity)), "subj with low wmc")
    thresh_speed = np.median(nback_speed)
    print(np.sum(nback_speed >= np.median(nback_speed)), "subj with high speed")
    print(np.sum(nback_speed < np.median(nback_speed)), "subj with low speed")
    thresh_accuracy = np.median(nback_accuracy)
    print(np.sum(nback_accuracy >= np.median(nback_accuracy)), "subj with high accuracy")
    print(np.sum(nback_accuracy < np.median(nback_accuracy)), "subj with low accuracy")
    thresh_impulsivity = np.median(nback_impulsivity)
    # thresh_impulsivity = nback_impulsivity[0]
    print(np.sum(nback_impulsivity >= np.median(nback_impulsivity)), "subj with high impulsivity")
    print(np.sum(nback_impulsivity < np.median(nback_impulsivity)), "subj with low impulsivity")
    # Convert
    nback_working_memory_capacity_class = (nback_working_memory_capacity >= thresh_wmc).astype(int)  # high: class 1, low: class 0
    nback_speed_class = (nback_speed >= thresh_speed).astype(int)
    nback_accuracy_class = (nback_accuracy >= thresh_accuracy).astype(int)
    # non_zero_values = nback_impulsivity[nback_impulsivity > 0]
    # thresh_impulsivity = np.percentile(non_zero_values, 50)
    nback_impulsivity_class = (nback_impulsivity >= thresh_impulsivity).astype(int)

    # Save classification values (all 4 in one array with column names and subject IDs)
    nback_all_class = pd.DataFrame({
        'subject_id': names_ret_nback,
        'working_memory_capacity': nback_working_memory_capacity_class,
        'speed': nback_speed_class,
        'accuracy': nback_accuracy_class,
        'impulsivity': nback_impulsivity_class
    })
    # nback_all_class.to_csv(path_save + "/nback_scores_classification_for_yasa_c3_eeg_rel_power_analysis.csv", index=False)

    return names_ret_nback


path_file = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs"
path_save = "/media/livia/Elements/public_sleep_data/stages/stages/original/yasa_eeg_powers"
dir_targets = sorted([file for file in glob2.glob(path_file + "/*/Modified_composite_all_*.xlsx") if not file.endswith("subset.xlsx")])

# found in "test_rnn_power.py"
path_file2 = r'/media/livia/Elements/public_sleep_data/stages/stages/original/'
subj_retained_for_power_analysis = pd.read_csv(path_file2 + "yasa_eeg_powers/subjects_retained_for_power_analysis.csv", header=None)
subj_retained_for_power_analysis = subj_retained_for_power_analysis.values.flatten().tolist()

# Read the targets and modify to only retain the subjects included in the yasa analysis and without too many nans (937)
method = 'subsets_separately'
# names_ret_nback = n_back_scores(dir_targets, subj_retained_for_power_analysis, method=method)

# Save the list of subjects retained for nback analysis
# names_ret_nback = pd.DataFrame(names_ret_nback)
# names_ret_nback.to_csv(path_save + "/subjects_retained_for_power_and_nback_analysis.csv", index=False)


def all_scores(dir_targets, subj_to_ret, method, composite_scores=True):

    if composite_scores:

        if method == 'subsets_separately':
            all_sustained_attention_score, all_working_memory_score, all_episodic_memory_score, \
                all_executive_functioning_score, all_overall_cognitive_score, subj_ids_all = np.array([]), \
                np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            for tar in range(len(dir_targets)):
                # tar = tar
                targets = pd.read_excel(dir_targets[tar])

                # List of columns to check for missing values
                columns_to_check = ["PVTB.PVTB_CR", "PVTB.PVTB_MEAN_RT", "PVTB.PVTB_355MS_LAP", "PVTB.PVTB_CERR",
                                    "SPCPTNL.SCPT_FP", "SPCPTNL.SCPT_TP", "SPCPTNL.SCPT_RT", "CPF_A.CPF_CR",
                                    "CPF_A.CPF_FP", "CPF_A.CPF_TPRT","PCET_A.PCET_CR", "PCET_A.PCET_CAT",
                                    "PCET_A.PCET_WIS_PER_ER", "PCET_A.PCET_WIS_CLEVEL_RES", "SLNB12_60.SLNB2_TP",
                                    "SLNB12_60.SLNB2_FP", "SLNB12_60.SLNB2_RTC"]
                # Find rows with non-missing values for all specified columns
                # Assuming 'subject_id' is the column name for subject IDs in your DataFrame
                targets1 = targets.dropna(subset=columns_to_check)

                # Get the subject IDs and index of rows that have all required values
                print("Number of all subjects in ", dir_targets[tar], "is: ", len(targets))
                subj_ids = targets1['test_sessions.subid'].to_numpy()
                print("Number of subjects with test scores in ", dir_targets[tar], "is: ", len(subj_ids))
                # indices_with_all_values = targets1.index.tolist()

                # Extract the desired scores and z-score them
                pvtb_valid_responses = zscore(targets1['PVTB.PVTB_CR']).to_numpy()
                pvtb_mean_rt = zscore(targets1['PVTB.PVTB_MEAN_RT']).to_numpy()
                pvtb_lapses_355ms = zscore(targets1['PVTB.PVTB_355MS_LAP']).to_numpy()
                pvtb_errors_commission = zscore(targets1['PVTB.PVTB_CERR']).to_numpy()
                spcptnl_tp_responses = zscore(targets1['SPCPTNL.SCPT_TP']).to_numpy()
                spcptnl_fp_responses = zscore(targets1['SPCPTNL.SCPT_FP']).to_numpy()
                spcptnl_median_rt = zscore(targets1['SPCPTNL.SCPT_RT']).to_numpy()
                slnb_true_positives = zscore(targets1['SLNB12_60.SLNB2_TP']).to_numpy()
                slnb_false_positives = zscore(targets1['SLNB12_60.SLNB2_FP']).to_numpy()
                slnb_median_rt_correct = zscore(targets1['SLNB12_60.SLNB2_RTC']).to_numpy()
                cpf_correct_responses = zscore(targets1['CPF_A.CPF_CR']).to_numpy()
                cpf_false_positives = zscore(targets1['CPF_A.CPF_FP']).to_numpy()
                cpf_median_rt_tp = zscore(targets1['CPF_A.CPF_TPRT']).to_numpy()
                pcet_correct_responses = zscore(targets1['PCET_A.PCET_CR']).to_numpy()
                pcet_categories_achieved = zscore(targets1['PCET_A.PCET_CAT']).to_numpy()
                pcet_perseverative_errors = zscore(targets1['PCET_A.PCET_WIS_PER_ER']).to_numpy()
                pcet_concept_level_responses = zscore(targets1['PCET_A.PCET_WIS_CLEVEL_RES']).to_numpy()

                # Compute the composite scores for each cognitive aspect by summing the z-scored values
                # (+ for good performance, - for bad)
                sustained_attention_score = pvtb_valid_responses - pvtb_mean_rt - pvtb_lapses_355ms - pvtb_errors_commission \
                                            + spcptnl_tp_responses - spcptnl_fp_responses - spcptnl_median_rt
                working_memory_score = slnb_true_positives - slnb_false_positives - slnb_median_rt_correct
                episodic_memory_score = cpf_correct_responses - cpf_false_positives - cpf_median_rt_tp
                executive_functioning_score = pcet_correct_responses + pcet_categories_achieved -\
                                              pcet_perseverative_errors + pcet_concept_level_responses
                overall_cognitive_score = sustained_attention_score + working_memory_score +\
                                          episodic_memory_score + executive_functioning_score

                # Concatenate the values from all the subsets
                all_sustained_attention_score = np.concatenate((all_sustained_attention_score, sustained_attention_score))
                all_working_memory_score = np.concatenate((all_working_memory_score, working_memory_score))
                all_episodic_memory_score = np.concatenate((all_episodic_memory_score, episodic_memory_score))
                all_executive_functioning_score = np.concatenate((all_executive_functioning_score, executive_functioning_score))
                all_overall_cognitive_score = np.concatenate((all_overall_cognitive_score, overall_cognitive_score))
                subj_ids_all = np.concatenate((subj_ids_all, subj_ids))

        elif method == 'all_subsets_together':

            # Initialize the arrays for all scores arrays to store the values
            pvtb_valid_responses_all, pvtb_mean_rt_all, pvtb_lapses_355ms_all, pvtb_errors_commission_all, \
                spcptnl_tp_responses_all, spcptnl_fp_responses_all, spcptnl_median_rt_all, slnb_true_positives_all, \
                slnb_false_positives_all, slnb_median_rt_correct_all, cpf_correct_responses_all, cpf_false_positives_all, \
                cpf_median_rt_tp_all, pcet_correct_responses_all, pcet_categories_achieved_all, pcet_perseverative_errors_all, \
                pcet_concept_level_responses_all, subj_ids_all = np.array([]), np.array([]), np.array([]), np.array([]), \
                np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), \
                np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            for tar in range(len(dir_targets)):

                targets = pd.read_excel(dir_targets[tar])

                # List of columns to check for missing values
                columns_to_check = ["PVTB.PVTB_CR", "PVTB.PVTB_MEAN_RT", "PVTB.PVTB_355MS_LAP", "PVTB.PVTB_CERR",
                                    "SPCPTNL.SCPT_FP", "SPCPTNL.SCPT_TP", "SPCPTNL.SCPT_RT", "CPF_A.CPF_CR",
                                    "CPF_A.CPF_FP", "CPF_A.CPF_TPRT","PCET_A.PCET_CR", "PCET_A.PCET_CAT",
                                    "PCET_A.PCET_WIS_PER_ER", "PCET_A.PCET_WIS_CLEVEL_RES", "SLNB12_60.SLNB2_TP",
                                    "SLNB12_60.SLNB2_FP", "SLNB12_60.SLNB2_RTC"]
                # Find rows with non-missing values for all specified columns
                # Assuming 'subject_id' is the column name for subject IDs in your DataFrame
                targets1 = targets.dropna(subset=columns_to_check)

                # Get the subject IDs and index of rows that have all required values
                print("Number of all subjects in ", dir_targets[tar], "is: ", len(targets))
                subj_ids = targets1['test_sessions.subid'].to_numpy()
                print("Number of subjects with test scores in ", dir_targets[tar], "is: ", len(subj_ids))
                # indices_with_all_values = targets1.index.tolist()

                # Extract the desired scores
                pvtb_valid_responses = targets1['PVTB.PVTB_CR'].to_numpy()
                pvtb_mean_rt = targets1['PVTB.PVTB_MEAN_RT'].to_numpy()
                pvtb_lapses_355ms = targets1['PVTB.PVTB_355MS_LAP'].to_numpy()
                pvtb_errors_commission = targets1['PVTB.PVTB_CERR'].to_numpy()
                spcptnl_tp_responses = targets1['SPCPTNL.SCPT_TP'].to_numpy()
                spcptnl_fp_responses = targets1['SPCPTNL.SCPT_FP'].to_numpy()
                spcptnl_median_rt = targets1['SPCPTNL.SCPT_RT'].to_numpy()
                slnb_true_positives = targets1['SLNB12_60.SLNB2_TP'].to_numpy()
                slnb_false_positives = targets1['SLNB12_60.SLNB2_FP'].to_numpy()
                slnb_median_rt_correct = targets1['SLNB12_60.SLNB2_RTC'].to_numpy()
                cpf_correct_responses = targets1['CPF_A.CPF_CR'].to_numpy()
                cpf_false_positives = targets1['CPF_A.CPF_FP'].to_numpy()
                cpf_median_rt_tp = targets1['CPF_A.CPF_TPRT'].to_numpy()
                pcet_correct_responses = targets1['PCET_A.PCET_CR'].to_numpy()
                pcet_categories_achieved = targets1['PCET_A.PCET_CAT'].to_numpy()
                pcet_perseverative_errors = targets1['PCET_A.PCET_WIS_PER_ER'].to_numpy()
                pcet_concept_level_responses = targets1['PCET_A.PCET_WIS_CLEVEL_RES'].to_numpy()

                # Concatenate the values from all the subsets
                pvtb_valid_responses_all = np.concatenate((pvtb_valid_responses_all, pvtb_valid_responses))
                pvtb_mean_rt_all = np.concatenate((pvtb_mean_rt_all, pvtb_mean_rt))
                pvtb_lapses_355ms_all = np.concatenate((pvtb_lapses_355ms_all, pvtb_lapses_355ms))
                pvtb_errors_commission_all = np.concatenate((pvtb_errors_commission_all, pvtb_errors_commission))
                spcptnl_tp_responses_all = np.concatenate((spcptnl_tp_responses_all, spcptnl_tp_responses))
                spcptnl_fp_responses_all = np.concatenate((spcptnl_fp_responses_all, spcptnl_fp_responses))
                spcptnl_median_rt_all = np.concatenate((spcptnl_median_rt_all, spcptnl_median_rt))
                slnb_true_positives_all = np.concatenate((slnb_true_positives_all, slnb_true_positives))
                slnb_false_positives_all = np.concatenate((slnb_false_positives_all, slnb_false_positives))
                slnb_median_rt_correct_all = np.concatenate((slnb_median_rt_correct_all, slnb_median_rt_correct))
                cpf_correct_responses_all = np.concatenate((cpf_correct_responses_all, cpf_correct_responses))
                cpf_false_positives_all = np.concatenate((cpf_false_positives_all, cpf_false_positives))
                cpf_median_rt_tp_all = np.concatenate((cpf_median_rt_tp_all, cpf_median_rt_tp))
                pcet_correct_responses_all = np.concatenate((pcet_correct_responses_all, pcet_correct_responses))
                pcet_categories_achieved_all = np.concatenate((pcet_categories_achieved_all, pcet_categories_achieved))
                pcet_perseverative_errors_all = np.concatenate((pcet_perseverative_errors_all, pcet_perseverative_errors))
                pcet_concept_level_responses_all = np.concatenate((pcet_concept_level_responses_all, pcet_concept_level_responses))
                subj_ids_all = np.concatenate((subj_ids_all, subj_ids))

            # Compute zscore of each score and then the composite scores for each cognitive aspect by
            # summing the z-scored values (+ for good performance, - for bad)
            all_sustained_attention_score = zscore(pvtb_valid_responses_all) - zscore(pvtb_mean_rt_all) - zscore(pvtb_lapses_355ms_all) - zscore(pvtb_errors_commission_all) \
                                            + zscore(spcptnl_tp_responses_all) - zscore(spcptnl_fp_responses_all) - zscore(spcptnl_median_rt_all)
            all_working_memory_score = zscore(slnb_true_positives_all) - zscore(slnb_false_positives_all) - zscore(slnb_median_rt_correct_all)
            all_episodic_memory_score = zscore(cpf_correct_responses_all) - zscore(cpf_false_positives_all) - zscore(cpf_median_rt_tp_all)
            all_executive_functioning_score = zscore(pcet_correct_responses_all) + zscore(pcet_categories_achieved_all) -\
                                                zscore(pcet_perseverative_errors_all) + zscore(pcet_concept_level_responses_all)
            all_overall_cognitive_score = all_sustained_attention_score + all_working_memory_score +\
                                            all_episodic_memory_score + all_executive_functioning_score

        names_ret = []
        for num_subj in range(len(subj_to_ret)):
            m = re.search('.*/([^/]+)$', subj_to_ret[num_subj])
            if m:
                name = m.group(1)
            # name = name.replace('/', '')
            names_ret.append(name)
        names_ret = np.array(names_ret)  # 935 (subjs retained for power analysis)

        sub_id_retain = [index for index, subject in enumerate(subj_ids_all) if subject in names_ret]  # 735 (subjs retained for nback analysis)
        names_ret_composite = [subject for subject in names_ret if subject in subj_ids_all]  # // //
        # missing_subjects = [subject for subject in names_ret if subject not in subj_ids_all]  # 200 (subjs from power analysis without nback test scores)

        # Retain only the subjects with scores
        all_sustained_attention_score = all_sustained_attention_score[sub_id_retain]
        all_working_memory_score = all_working_memory_score[sub_id_retain]
        all_episodic_memory_score = all_episodic_memory_score[sub_id_retain]
        all_executive_functioning_score = all_executive_functioning_score[sub_id_retain]
        all_overall_cognitive_score = all_overall_cognitive_score[sub_id_retain]

        # Save regression values (all 5 in one array with column names and subject IDs)
        all_scores = pd.DataFrame({
            'subject_id': names_ret_composite,
            'sustained_attention': all_sustained_attention_score,
            'working_memory': all_working_memory_score,
            'episodic_memory': all_episodic_memory_score,
            'executive_functioning': all_executive_functioning_score,
            'overall_cognitive': all_overall_cognitive_score
        })
        all_scores.to_csv(path_save + "/all_composite_scores_regression_for_yasa_c3_eeg_rel_power_analysis2.csv", index=False)

        # Convert regression to binary classification for each cognitive aspect using the median as threshold
        thresh_sustained_attention = np.median(all_sustained_attention_score)
        print(np.sum(all_sustained_attention_score >= np.median(all_sustained_attention_score)), "subj with high sustained attention")
        print(np.sum(all_sustained_attention_score < np.median(all_sustained_attention_score)), "subj with low sustained attention")
        thresh_working_memory = np.median(all_working_memory_score)
        print(np.sum(all_working_memory_score >= np.median(all_working_memory_score)), "subj with high working memory")
        print(np.sum(all_working_memory_score < np.median(all_working_memory_score)), "subj with low working memory")
        thresh_episodic_memory = np.median(all_episodic_memory_score)
        print(np.sum(all_episodic_memory_score >= np.median(all_episodic_memory_score)), "subj with high episodic memory")
        print(np.sum(all_episodic_memory_score < np.median(all_episodic_memory_score)), "subj with low episodic memory")
        thresh_executive_functioning = np.median(all_executive_functioning_score)
        print(np.sum(all_executive_functioning_score >= np.median(all_executive_functioning_score)), "subj with high executive functioning")
        print(np.sum(all_executive_functioning_score < np.median(all_executive_functioning_score)), "subj with low executive functioning")
        thresh_overall_cognitive_score = np.median(all_overall_cognitive_score)
        print(np.sum(all_overall_cognitive_score >= np.median(all_overall_cognitive_score)), "subj with high overall cognitive score")
        print(np.sum(all_overall_cognitive_score < np.median(all_overall_cognitive_score)), "subj with low overall cognitive score")
        # Convert
        all_sustained_attention_class = (all_sustained_attention_score >= thresh_sustained_attention).astype(int)  # high: class 1, low: class 0
        all_working_memory_class = (all_working_memory_score >= thresh_working_memory).astype(int)
        all_episodic_memory_class = (all_episodic_memory_score >= thresh_episodic_memory).astype(int)
        all_executive_functioning_class = (all_executive_functioning_score >= thresh_executive_functioning).astype(int)
        all_overall_cognitive_score_class = (all_overall_cognitive_score >= thresh_overall_cognitive_score).astype(int)

        # Save classification values (all 5 in one array with column names and subject IDs)
        all_scores_class = pd.DataFrame({
            'subject_id': names_ret_composite,
            'sustained_attention': all_sustained_attention_class,
            'working_memory': all_working_memory_class,
            'episodic_memory': all_episodic_memory_class,
            'executive_functioning': all_executive_functioning_class,
            'overall_cognitive': all_overall_cognitive_score_class
        })
        all_scores_class.to_csv(path_save + "/all_composite_scores_classification_for_yasa_c3_eeg_rel_power_analysis2.csv", index=False)

        return names_ret_composite

    else:
        # Initialize the arrays for all scores arrays to store the values
        pvtb_valid_responses_all, pvtb_mean_rt_all, pvtb_lapses_355ms_all, pvtb_errors_commission_all, \
            spcptnl_tp_responses_all, spcptnl_fp_responses_all, spcptnl_median_rt_all, slnb_true_positives_all, \
            slnb_false_positives_all, slnb_median_rt_correct_all, cpf_correct_responses_all, cpf_false_positives_all, \
            cpf_median_rt_tp_all, pcet_correct_responses_all, pcet_categories_achieved_all, pcet_perseverative_errors_all, \
            pcet_concept_level_responses_all, subj_ids_all = np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array(
            []), \
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        for tar in range(len(dir_targets)):
            targets = pd.read_excel(dir_targets[tar])

            # List of columns to check for missing values
            columns_to_check = ["PVTB.PVTB_CR", "PVTB.PVTB_MEAN_RT", "PVTB.PVTB_355MS_LAP", "PVTB.PVTB_CERR",
                                "SPCPTNL.SCPT_FP", "SPCPTNL.SCPT_TP", "SPCPTNL.SCPT_RT", "CPF_A.CPF_CR",
                                "CPF_A.CPF_FP", "CPF_A.CPF_TPRT", "PCET_A.PCET_CR", "PCET_A.PCET_CAT",
                                "PCET_A.PCET_WIS_PER_ER", "PCET_A.PCET_WIS_CLEVEL_RES", "SLNB12_60.SLNB2_TP",
                                "SLNB12_60.SLNB2_FP", "SLNB12_60.SLNB2_RTC"]
            # Find rows with non-missing values for all specified columns
            # Assuming 'subject_id' is the column name for subject IDs in your DataFrame
            # targets1 = targets.dropna(subset=columns_to_check)
            targets1 = targets

            # Get the subject IDs and index of rows that have all required values
            print("Number of all subjects in ", dir_targets[tar], "is: ", len(targets))
            subj_ids = targets1['test_sessions.subid'].to_numpy()
            print("Number of subjects with test scores in ", dir_targets[tar], "is: ", len(subj_ids))
            # indices_with_all_values = targets1.index.tolist()

            # Extract the desired scores
            pvtb_valid_responses = targets1['PVTB.PVTB_CR'].to_numpy()
            pvtb_mean_rt = targets1['PVTB.PVTB_MEAN_RT'].to_numpy()
            pvtb_lapses_355ms = targets1['PVTB.PVTB_355MS_LAP'].to_numpy()
            pvtb_errors_commission = targets1['PVTB.PVTB_CERR'].to_numpy()
            spcptnl_tp_responses = targets1['SPCPTNL.SCPT_TP'].to_numpy()
            spcptnl_fp_responses = targets1['SPCPTNL.SCPT_FP'].to_numpy()
            spcptnl_median_rt = targets1['SPCPTNL.SCPT_RT'].to_numpy()
            slnb_true_positives = targets1['SLNB12_60.SLNB2_TP'].to_numpy()
            # slnb_false_positives = targets1['SLNB12_60.SLNB2_FP'].to_numpy()
            slnb_median_rt_correct = targets1['SLNB12_60.SLNB2_RTC'].to_numpy()
            cpf_correct_responses = targets1['CPF_A.CPF_CR'].to_numpy()
            cpf_false_positives = targets1['CPF_A.CPF_FP'].to_numpy()
            cpf_median_rt_tp = targets1['CPF_A.CPF_TPRT'].to_numpy()
            pcet_correct_responses = targets1['PCET_A.PCET_CR'].to_numpy()
            pcet_categories_achieved = targets1['PCET_A.PCET_CAT'].to_numpy()
            pcet_perseverative_errors = targets1['PCET_A.PCET_WIS_PER_ER'].to_numpy()
            pcet_concept_level_responses = targets1['PCET_A.PCET_WIS_CLEVEL_RES'].to_numpy()

            # Concatenate the values from all the subsets
            pvtb_valid_responses_all = np.concatenate((pvtb_valid_responses_all, pvtb_valid_responses))
            pvtb_mean_rt_all = np.concatenate((pvtb_mean_rt_all, pvtb_mean_rt))
            pvtb_lapses_355ms_all = np.concatenate((pvtb_lapses_355ms_all, pvtb_lapses_355ms))
            pvtb_errors_commission_all = np.concatenate((pvtb_errors_commission_all, pvtb_errors_commission))
            spcptnl_tp_responses_all = np.concatenate((spcptnl_tp_responses_all, spcptnl_tp_responses))
            spcptnl_fp_responses_all = np.concatenate((spcptnl_fp_responses_all, spcptnl_fp_responses))
            spcptnl_median_rt_all = np.concatenate((spcptnl_median_rt_all, spcptnl_median_rt))
            slnb_true_positives_all = np.concatenate((slnb_true_positives_all, slnb_true_positives))
            # slnb_false_positives_all = np.concatenate((slnb_false_positives_all, slnb_false_positives))
            slnb_median_rt_correct_all = np.concatenate((slnb_median_rt_correct_all, slnb_median_rt_correct))
            cpf_correct_responses_all = np.concatenate((cpf_correct_responses_all, cpf_correct_responses))
            cpf_false_positives_all = np.concatenate((cpf_false_positives_all, cpf_false_positives))
            cpf_median_rt_tp_all = np.concatenate((cpf_median_rt_tp_all, cpf_median_rt_tp))
            pcet_correct_responses_all = np.concatenate((pcet_correct_responses_all, pcet_correct_responses))
            pcet_categories_achieved_all = np.concatenate((pcet_categories_achieved_all, pcet_categories_achieved))
            pcet_perseverative_errors_all = np.concatenate((pcet_perseverative_errors_all, pcet_perseverative_errors))
            pcet_concept_level_responses_all = np.concatenate(
                (pcet_concept_level_responses_all, pcet_concept_level_responses))
            subj_ids_all = np.concatenate((subj_ids_all, subj_ids))

        # Compute zscore of each score
        pvtb_valid_responses_all = zscore(pvtb_valid_responses_all, nan_policy='omit')
        pvtb_mean_rt_all = zscore(pvtb_mean_rt_all, nan_policy='omit')
        pvtb_lapses_355ms_all = zscore(pvtb_lapses_355ms_all, nan_policy='omit')
        pvtb_errors_commission_all = zscore(pvtb_errors_commission_all, nan_policy='omit')
        spcptnl_tp_responses_all = zscore(spcptnl_tp_responses_all, nan_policy='omit')
        spcptnl_fp_responses_all = zscore(spcptnl_fp_responses_all, nan_policy='omit')
        spcptnl_median_rt_all = zscore(spcptnl_median_rt_all, nan_policy='omit')
        slnb_true_positives_all = zscore(slnb_true_positives_all, nan_policy='omit')
        # slnb_false_positives_all = zscore(slnb_false_positives_all, nan_policy='omit')
        slnb_median_rt_correct_all = zscore(slnb_median_rt_correct_all, nan_policy='omit')
        cpf_correct_responses_all = zscore(cpf_correct_responses_all, nan_policy='omit')
        cpf_false_positives_all = zscore(cpf_false_positives_all, nan_policy='omit')
        cpf_median_rt_tp_all = zscore(cpf_median_rt_tp_all, nan_policy='omit')
        pcet_correct_responses_all = zscore(pcet_correct_responses_all, nan_policy='omit')
        pcet_categories_achieved_all = zscore(pcet_categories_achieved_all, nan_policy='omit')
        pcet_perseverative_errors_all = zscore(pcet_perseverative_errors_all, nan_policy='omit')
        pcet_concept_level_responses_all = zscore(pcet_concept_level_responses_all, nan_policy='omit')

        # # Save regression values (all in one array with column names and subject IDs)
        all_scores = pd.DataFrame({
            'subject_id': subj_ids_all,
            'pvtb_valid_responses': pvtb_valid_responses_all,
            'pvtb_mean_rt': pvtb_mean_rt_all,
            'pvtb_lapses_355ms': pvtb_lapses_355ms_all,
            'pvtb_errors_commission': pvtb_errors_commission_all,
            'spcptnl_tp_responses': spcptnl_tp_responses_all,
            'spcptnl_fp_responses': spcptnl_fp_responses_all,
            'spcptnl_median_rt': spcptnl_median_rt_all,
            'slnb_true_positives': slnb_true_positives_all,
            # 'slnb_false_positives': slnb_false_positives_all,
            'slnb_median_rt_correct': slnb_median_rt_correct_all,
            'cpf_correct_responses': cpf_correct_responses_all,
            'cpf_false_positives': cpf_false_positives_all,
            'cpf_median_rt_tp': cpf_median_rt_tp_all,
            'pcet_correct_responses': pcet_correct_responses_all,
            'pcet_categories_achieved': pcet_categories_achieved_all,
            'pcet_perseverative_errors': pcet_perseverative_errors_all,
            'pcet_concept_level_responses': pcet_concept_level_responses_all
        })
        all_scores.to_csv(path_save + "/all_scores_regression_for_yasa_c3_eeg_rel_power_analysis.csv", index=False)

        # Convert regression to binary classification for each score using the median as threshold
        thresh_pvtb_valid_responses = np.nanmedian(pvtb_valid_responses_all)
        print(np.sum(pvtb_valid_responses_all >= thresh_pvtb_valid_responses), "subj with high pvtb_valid_responses")
        print(np.sum(pvtb_valid_responses_all < thresh_pvtb_valid_responses), "subj with low pvtb_valid_responses")
        thresh_pvtb_mean_rt = np.nanmedian(pvtb_mean_rt_all)
        print(np.sum(pvtb_mean_rt_all >= thresh_pvtb_mean_rt), "subj with high pvtb_mean_rt")
        print(np.sum(pvtb_mean_rt_all < thresh_pvtb_mean_rt), "subj with low pvtb_mean_rt")
        thresh_pvtb_lapses_355ms = np.nanmedian(pvtb_lapses_355ms_all)
        print(np.sum(pvtb_lapses_355ms_all >= thresh_pvtb_lapses_355ms), "subj with high pvtb_lapses_355ms")
        print(np.sum(pvtb_lapses_355ms_all < thresh_pvtb_lapses_355ms), "subj with low pvtb_lapses_355ms")
        thresh_pvtb_errors_commission = np.nanmedian(pvtb_errors_commission_all)
        print(np.sum(pvtb_errors_commission_all >= thresh_pvtb_errors_commission), "subj with high pvtb_errors_commission")
        print(np.sum(pvtb_errors_commission_all < thresh_pvtb_errors_commission), "subj with low pvb_errors_commission")
        thresh_spcptnl_tp_responses = np.nanmedian(spcptnl_tp_responses_all)
        print(np.sum(spcptnl_tp_responses_all >= thresh_spcptnl_tp_responses), "subj with high spcptnl_tp_responses")
        print(np.sum(spcptnl_tp_responses_all < thresh_spcptnl_tp_responses), "subj with low spcptnl_tp_responses")
        thresh_spcptnl_fp_responses = np.nanmedian(spcptnl_fp_responses_all)
        print(np.sum(spcptnl_fp_responses_all >= thresh_spcptnl_fp_responses), "subj with high spcptnl_fp_responses")
        print(np.sum(spcptnl_fp_responses_all < thresh_spcptnl_fp_responses), "subj with low spcptnl_fp_responses")
        thresh_spcptnl_median_rt = np.nanmedian(spcptnl_median_rt_all)
        print(np.sum(spcptnl_median_rt_all >= thresh_spcptnl_median_rt), "subj with high spcptnl_median_rt")
        print(np.sum(spcptnl_median_rt_all < thresh_spcptnl_median_rt), "subj with low spcptnl_median_rt")
        thresh_slnb_true_positives = np.nanmedian(slnb_true_positives_all)
        print(np.sum(slnb_true_positives_all >= thresh_slnb_true_positives), "subj with high slnb_true_positives")
        print(np.sum(slnb_true_positives_all < thresh_slnb_true_positives), "subj with low slnb_true_positives")
        # thresh_slnb_false_positives = np.nanmedian(slnb_false_positives_all)
        # print(np.sum(slnb_false_positives_all >= thresh_slnb_false_positives), "subj with high slnb_false_positives")
        # print(np.sum(slnb_false_positives_all < thresh_slnb_false_positives), "subj with low slnb_false_positives")
        thresh_slnb_median_rt_correct = np.nanmedian(slnb_median_rt_correct_all)
        print(np.sum(slnb_median_rt_correct_all >= thresh_slnb_median_rt_correct), "subj with high slnb_median_rt_correct")
        print(np.sum(slnb_median_rt_correct_all < thresh_slnb_median_rt_correct), "subj with low slnb_median_rt_correct")
        thresh_cpf_correct_responses = np.nanmedian(cpf_correct_responses_all)
        print(np.sum(cpf_correct_responses_all >= thresh_cpf_correct_responses), "subj with high cpf_correct_responses")
        print(np.sum(cpf_correct_responses_all < thresh_cpf_correct_responses), "subj with low cpf_correct_responses")
        thresh_cpf_false_positives = np.nanmedian(cpf_false_positives_all)
        print(np.sum(cpf_false_positives_all >= thresh_cpf_false_positives), "subj with high cpf_false_positives")
        print(np.sum(cpf_false_positives_all < thresh_cpf_false_positives), "subj with low cpf_false_positives")
        thresh_cpf_median_rt_tp = np.nanmedian(cpf_median_rt_tp_all)
        print(np.sum(cpf_median_rt_tp_all >= thresh_cpf_median_rt_tp), "subj with high cpf_median_rt_tp")
        print(np.sum(cpf_median_rt_tp_all < thresh_cpf_median_rt_tp), "subj with low cpf_median_rt_tp")
        thresh_pcet_correct_responses = np.nanmedian(pcet_correct_responses_all)
        print(np.sum(pcet_correct_responses_all >= thresh_pcet_correct_responses), "subj with high pcet_correct_responses")
        print(np.sum(pcet_correct_responses_all < thresh_pcet_correct_responses), "subj with low pcet_correct_responses")
        thresh_pcet_categories_achieved = np.nanmedian(pcet_categories_achieved_all)
        print(np.sum(pcet_categories_achieved_all >= thresh_pcet_categories_achieved), "subj with high pcet_categories_achieved")
        print(np.sum(pcet_categories_achieved_all < thresh_pcet_categories_achieved), "subj with low pcet_categories_achieved")
        thresh_pcet_perseverative_errors = np.nanmedian(pcet_perseverative_errors_all)
        print(np.sum(pcet_perseverative_errors_all >= thresh_pcet_perseverative_errors), "subj with high pcet_perseverative_errors")
        print(np.sum(pcet_perseverative_errors_all < thresh_pcet_perseverative_errors), "subj with low pcet_perseverative_errors")
        thresh_pcet_concept_level_responses = np.nanmedian(pcet_concept_level_responses_all)
        print(np.sum(pcet_concept_level_responses_all >= thresh_pcet_concept_level_responses), "subj with high pcet_concept_level_responses")
        print(np.sum(pcet_concept_level_responses_all < thresh_pcet_concept_level_responses), "subj with low pcet_concept_level_responses")

        # Regression to classification with NaN preservation
        def classify_with_nan(data, threshold):
            return np.where(
                np.isnan(data),  # Check for NaN
                np.nan,  # Preserve NaN
                (data >= threshold).astype(int)  # Classification
            )

        # Convert
        pvtb_valid_responses_class = classify_with_nan(pvtb_valid_responses_all, thresh_pvtb_valid_responses)
        pvtb_mean_rt_class = classify_with_nan(pvtb_mean_rt_all, thresh_pvtb_mean_rt)
        pvtb_lapses_355ms_class = classify_with_nan(pvtb_lapses_355ms_all, thresh_pvtb_lapses_355ms)
        pvtb_errors_commission_class = classify_with_nan(pvtb_errors_commission_all, thresh_pvtb_errors_commission)
        spcptnl_tp_responses_class = classify_with_nan(spcptnl_tp_responses_all, thresh_spcptnl_tp_responses)
        spcptnl_fp_responses_class = classify_with_nan(spcptnl_fp_responses_all, thresh_spcptnl_fp_responses)
        spcptnl_median_rt_class = classify_with_nan(spcptnl_median_rt_all, thresh_spcptnl_median_rt)
        slnb_true_positives_class = classify_with_nan(slnb_true_positives_all, thresh_slnb_true_positives)
        # slnb_false_positives_class = classify_with_nan(slnb_false_positives_all, thresh_slnb_false_positives)
        slnb_median_rt_correct_class = classify_with_nan(slnb_median_rt_correct_all, thresh_slnb_median_rt_correct)
        cpf_correct_responses_class = classify_with_nan(cpf_correct_responses_all, thresh_cpf_correct_responses)
        cpf_false_positives_class = classify_with_nan(cpf_false_positives_all, thresh_cpf_false_positives)
        cpf_median_rt_tp_class = classify_with_nan(cpf_median_rt_tp_all, thresh_cpf_median_rt_tp)
        pcet_correct_responses_class = classify_with_nan(pcet_correct_responses_all, thresh_pcet_correct_responses)
        pcet_categories_achieved_class = classify_with_nan(pcet_categories_achieved_all, thresh_pcet_categories_achieved)
        pcet_perseverative_errors_class = classify_with_nan(pcet_perseverative_errors_all, thresh_pcet_perseverative_errors)
        pcet_concept_level_responses_class = classify_with_nan(pcet_concept_level_responses_all, thresh_pcet_concept_level_responses)

        # Save classification values (all in one array with column names and subject IDs)
        all_scores_class = pd.DataFrame({
            'subject_id': subj_ids_all,
            'pvtb_valid_responses': pvtb_valid_responses_class,
            'pvtb_mean_rt': pvtb_mean_rt_class,
            'pvtb_lapses_355ms': pvtb_lapses_355ms_class,
            'pvtb_errors_commission': pvtb_errors_commission_class,
            'spcptnl_tp_responses': spcptnl_tp_responses_class,
            'spcptnl_fp_responses': spcptnl_fp_responses_class,
            'spcptnl_median_rt': spcptnl_median_rt_class,
            'slnb_true_positives': slnb_true_positives_class,
            # 'slnb_false_positives': slnb_false_positives_class,
            'slnb_median_rt_correct': slnb_median_rt_correct_class,
            'cpf_correct_responses': cpf_correct_responses_class,
            'cpf_false_positives': cpf_false_positives_class,
            'cpf_median_rt_tp': cpf_median_rt_tp_class,
            'pcet_correct_responses': pcet_correct_responses_class,
            'pcet_categories_achieved': pcet_categories_achieved_class,
            'pcet_perseverative_errors': pcet_perseverative_errors_class,
            'pcet_concept_level_responses': pcet_concept_level_responses_class
        })
        all_scores_class.to_csv(path_save + "/all_scores_classification_for_yasa_c3_eeg_rel_power_analysis.csv", index=False)


method = 'all_subsets_together'
cp = False
names_ret_composite_all = all_scores(dir_targets, subj_retained_for_power_analysis, method=method, composite_scores=cp)

# Save the list of subjects retained for nback analysis
names_ret_composite_all = pd.DataFrame(names_ret_composite_all)
names_ret_composite_all.to_csv(path_save + "/subjects_retained_for_power_and_composite_all_analysis.csv", index=False)

a = 0
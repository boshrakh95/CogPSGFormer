
# Process and save ECG HRV features from the STAGES dataset
# Sleep stages not considered (still do not have reliable hypnograms)
# 30 Oct 2024
# Boshra

# To do:
# a large part of the beginning of most of the ECG signals (and the ending for some too) is just noise (incorrect rpeak)


import mne
import yasa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import glob2
import os
import re
# from scipy.signal import welch
# from mne.time_frequency import psd_array_multitaper
import neurokit2 as nk
from scipy.signal import decimate

matplotlib.use('Agg')

# All possible ECG1 channel names in the STAGES dataset
ecg_channel_list = ['EKG', 'EKG__1', 'EKG1', 'ECG', 'ECG_I', 'ECG1', 'ECG_1', 'EKG_#1']  # 'C3M2' for visualization


# Read the EDF file and retain all the ECG channels by searching through "ecg_channel_list"
def retain_ecg_channels_from_list(edf_file_path, ecg_channel_list):

    # Read EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=False)

    # Get the list of all available channels in the file
    available_channels = raw.info['ch_names']

    # Find the intersection of the available channels and the EEG channel list
    ecg_channels_to_retain = [ch for ch in available_channels if ch in ecg_channel_list]

    if ecg_channels_to_retain:
        raw = mne.io.read_raw_edf(edf_file_path, preload=True, include=ecg_channels_to_retain)
        sf = raw.info['sfreq']
    else:
        ecg_channels_to_retain = [ch for ch in available_channels if ch in ['ECG_2', 'ECG_II']]
        raw = mne.io.read_raw_edf(edf_file_path, preload=True, include=ecg_channels_to_retain)
        sf = raw.info['sfreq']

    return raw, int(sf), ecg_channels_to_retain if raw else None


path_file = r'/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/'
path_file2 = r'/media/livia/Elements/public_sleep_data/stages/stages/original/'
clinics = glob2.glob(path_file + "*")
# clinics = clinics[6:]

for clinic in clinics:

    print("Clinic ", clinic)
    data_path = os.path.join(path_file, clinic, 'usable')
    dir_subjs = sorted(glob2.glob(data_path + "/*.edf"))
    directory1 = "yasa_outputs/"
    path_par1 = os.path.join(path_file, clinic, directory1)
    directory2 = "mask_yasa_5sec_originalfs_0.3to35Hz/"
    path_mask = os.path.join(path_par1, directory2)
    mask_files = sorted(glob2.glob(path_mask + "/*.csv"))
    subj_retained_for_power_analysis = pd.read_csv(path_file2 + "yasa_eeg_powers/subjects_retained_for_power_analysis.csv", header=None)
    subj_retained_for_power_analysis = subj_retained_for_power_analysis.values.flatten().tolist()
    directory3 = "ecg_segmented_2min/"
    path_save = os.path.join(path_par1, directory3)
    os.makedirs(path_save, exist_ok=True)

    # extract all the subject IDs
    names = []
    for num_subj in range(len(dir_subjs)):
        m = re.search('usable/(.+?).edf', dir_subjs[num_subj])
        if m:
            name = m.group(1)
        # name = name.replace('/', '')
        names.append(name)
    names = np.array(names)

    # Update dir_subjs to only include the final cohort for power analysis (with available/valid eeg power features)
    names_ret = []
    for num_subj in range(len(subj_retained_for_power_analysis)):
        m = re.search('.*/([^/]+)$', subj_retained_for_power_analysis[num_subj])
        if m:
            name = m.group(1)
        # name = name.replace('/', '')
        names_ret.append(name)
    names_ret = np.array(names_ret)
    # Step 1: Find mutual elements using intersection
    mutual_elements = np.intersect1d(names, names_ret)
    indexes1 = [np.where(names == element)[0][0] for element in mutual_elements]
    dir_subjs = np.array(dir_subjs)[indexes1].tolist()
    mask_files = np.array(mask_files)[indexes1].tolist()

    # retained_eeg_channels = []
    # samp_freq = []
    if len(dir_subjs) > 0:
        channels_used = []
        for subject in range(len(dir_subjs)):
            # subject = subject + 59

            print('Start processing', mutual_elements[subject]+'.edf ...')
            raw = retain_ecg_channels_from_list(dir_subjs[subject], ecg_channel_list)

            if raw:
                raw_channel, sf, channels = raw
                channels_used.append(np.array(channels))
                print("ECG signal found:")

                ecg_signal = raw_channel.get_data()
                ecg_signal = ecg_signal[0, :].flatten()
                ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sf, method="biosppy", highpass=0.3, lowpass=45)

                # Plotting original and cleaned ECG signals
                # matplotlib.use('TkAgg')
                # plt.figure(figsize=(12, 6))
                # plt.subplot(2, 1, 1)
                # plt.plot(ecg_signal, label="Original ECG", color="blue")
                # plt.title("Original ECG Signal")
                # plt.subplot(2, 1, 2)
                # plt.plot(ecg_cleaned, label="Cleaned ECG", color="green")
                # plt.title("Cleaned ECG Signal")
                # plt.tight_layout()
                # plt.show()

                # Optional - Assess Signal Quality and Remove Poor Quality Segments
                # quality = nk.ecg_quality(ecg_cleaned, sampling_rate=sf)
                # clean_segments = ecg_cleaned[quality >= 0.5]

                # Load and apply 5-sec yasa masks (originally created using yasa.art_detect() on EEG)
                mask = np.loadtxt(mask_files[subject], delimiter=',', skiprows=0, dtype=float)
                mask_up = yasa.hypno_upsample_to_data(hypno=mask, sf_hypno=(1 / 5), data=ecg_cleaned, sf_data=sf)
                print(mask_up.size == ecg_cleaned.shape[0])  # Does the hypnogram have the same number of samples as data?
                where_clean = (mask_up == 0)  # True if sample is clean / False if it belongs to an artifactual 5-sec
                ecg_cleaned = ecg_cleaned[where_clean]
                print("Clean data shape:", ecg_cleaned.shape)

                # Visualize eeg masks on top of ecg signals to see if it's sensible to use them
                # matplotlib.use('TkAgg')
                # plt.figure()
                # plt.plot(ecg_cleaned)
                # plt.plot(mask_up*max(ecg_cleaned)/2)
                # plt.show()

                # Downsample to 90 Hz
                sf_old = sf
                sf = 90
                decimation_factor = int(sf_old / sf)
                ecg_cleaned = decimate(ecg_cleaned, q=decimation_factor, zero_phase=True)

                # Detect R-Peaks
                r_peaks = nk.ecg_findpeaks(ecg_cleaned, sampling_rate=sf, method="neurokit")
                r_peak_indices = r_peaks["ECG_R_Peaks"]

                # r_peaks2 = nk.ecg_findpeaks(ecg_cleaned, sampling_rate=sf, method="manikandan2012")
                # r_peak_indices2 = r_peaks2["ECG_R_Peaks"]

                # Plot the ECG signal with detected R-peaks to compare different methods
                # matplotlib.use('TkAgg')
                # plt.figure(figsize=(12, 8))
                # plt.subplot(2, 1, 1)
                # time = np.arange(len(ecg_cleaned)) / sf / 60  # in minute
                # plt.plot(time, ecg_cleaned, label="Cleaned ECG", color="blue")
                # plt.scatter(r_peak_indices/sf/60, ecg_cleaned[r_peak_indices], color="red",
                #             label="R-peaks (neurokit)", marker="o")
                # # plt.plot(time, mask_up*max(ecg_cleaned)/2, label="Mask", color="black")
                # # plt.title("R-Peak Detection - Method: Neurokit")
                # plt.subplot(2, 1, 2)
                # plt.plot(time, eeg_cleaned, label="Cleaned ECG", color="blue")
                # # plt.plot(time, mask_up * max(eeg_cleaned) / 2, label="Mask", color="black")
                # # plt.scatter(r_peak_indices2, ecg_cleaned[r_peak_indices2], color="green",
                # #             label="R-peaks (pantompkins1985)", marker="o")
                # # plt.title("R-Peak Detection - Method: Pan-Tompkins")
                # plt.tight_layout()
                # plt.show()

                # Segment all the channels into "window" seconds
                window_time_t = 2  # min
                window_size_t = window_time_t * 60 * sf  # For a 2-minute window in samples
                step_size_t = window_size_t  # no overlap  # int(window_size_t // 2)  # 50% overlap
                # step_size_t = int(window_size_t * 0.7)  # 30% overlap
                num_segments_t = int((len(ecg_cleaned) - window_size_t) // step_size_t + 1)
                print("Number of segments for time analysis:", num_segments_t)

                seg_window = 30*4  # 2 minutes
                _, data_clean_segmented = yasa.sliding_window(ecg_cleaned, sf, window=seg_window)
                print(f"Data shape after cleaning and {seg_window}-sec segmentation (epochs, samples):",
                      data_clean_segmented.shape)

                print(data_clean_segmented.shape[0] == num_segments_t)  # Check if the number of segments is correct

                min_rpeak_t = 80
                max_rpeak_t = 200
                index_rem = []
                for i in range(num_segments_t):
                    # Define the segment start and end in samples
                    start = i * step_size_t
                    end = start + window_size_t

                    # Find R-peaks within this segment
                    r_peaks_segment = r_peak_indices[(r_peak_indices >= start) & (r_peak_indices < end)]

                    # Keep track of segment indexes with unreasonable R-peak counts (that should be removed)
                    if not (min_rpeak_t <= len(r_peaks_segment) <= max_rpeak_t):
                        index_rem.append(i)

                # Remove segments with unreasonable R-peak counts
                data_clean_segmented = np.delete(data_clean_segmented, index_rem, axis=0)
                print("Data shape after removing segments with unreasonable R-peak counts (epochs, num_samples):",
                      data_clean_segmented.shape)

                # matplotlib.use('TkAgg')
                # a = bp[:, 3, :]  # select one channel, a contains the power of the 6 bands for each epoch
                # plt.figure(figsize=(10, 6))
                # sns.heatmap(data, cmap='viridis', annot=False)  # You can change 'viridis' to any other colormap
                # plt.show()

                # Save the segmented data in a numpy file
                path_sbj = os.path.join(path_save, mutual_elements[subject])
                os.makedirs(path_sbj, exist_ok=True)
                np.save(os.path.join(path_sbj, 'ecg_segmented_2min.npy'), data_clean_segmented)
            else:
                print("Required channels not found!")

        # df1 = pd.DataFrame(channels_used)
        # df1.to_csv(path_par1 + 'ecg_channel_used_for_hrv_analysis.csv', index=False, header=False)

        aa=0


# =======================================================================================================================

# Define the paths to both directories
dir1 = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/STNF/yasa_outputs/ecg_hrv_params"
dir2 = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/STNF/yasa_outputs/eeg_bandpowers_30sec"

# Get the list of folders in each directory
folders_dir1 = set(os.listdir(dir1))
folders_dir2 = set(os.listdir(dir2))

# Find the missing folder by comparing sets
missing_folder = folders_dir1 - folders_dir2 if len(folders_dir1) > len(folders_dir2) else folders_dir2 - folders_dir1

print("Missing folder:", missing_folder)







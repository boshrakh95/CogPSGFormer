
# Extract some main EEG spectral features from the STAGES dataset
# Sleep stages not considered (still do not have reliable hypnograms)
# 22 Oct 2024
# Boshra

# To do:
# For best performance, apply yasa.art_detect on pre-staged data and make sure to pass the hypnogram.
# Sleep stages have very different EEG signatures and the artifect rejection will be much more accurate when applied separately on each sleep stage.

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
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper

matplotlib.use('Agg')

# All possible EEG channel names in the STAGES dataset
eeg_channel_list = ['E1M2', 'E2M2', 'C3M2', 'C4M1', 'O1M2', 'O2M1',
                    'F3M2', 'F4M1', 'F1M2', 'F2M1', 'T3M2', 'T4M1', 'Pz', 'P4', 'EEG_F3_A2', 'EEG_F4_A1', 'EEG_A1_A2',
                    'EEG_C3_A2', 'EEG_C4_A1', 'EEG_O1_A2', 'EEG_O2_A1', 'EEG_T3_A2', 'EEG_T4_A1', 'EEG_P3_A2',
                    'EEG_P4_A1', 'Fpz', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'Oz', 'O1', 'O2', 'M1', 'M2',
                    'Fz', 'Cz', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'EEG_Fp1_A2', 'EEG_Fp2_A1', 'EEG_Fp1_A22',
                    'EEG_Fp2_A12', 'EEG_F3_A22', 'EEG_F4_A12', 'EEG_A1_A22', 'EEG_C3_A22', 'EEG_C4_A12', 'EEG_O1_A22',
                    'EEG_O2_A12', 'EEG_F3_A1', 'EEG_F4_A2', 'EEG_C3_A1', 'EEG_C4_A2',
                    'EEG_O1_A1', 'EEG_O2_A2', 'EEG_Fp1_A22', 'E1', 'E2', 'EEG_F3-A2', 'EEG_F4-A1',
                    'EEG_A1-A2', 'EEG_C3-A2', 'EEG_C4-A1', 'EEG_O1-A2', 'EEG_O2-A1']


# Read the EDF file and retain all the EEG channels by searching through "eeg_channel_list"
def retain_eeg_channels_from_list(edf_file_path, eeg_channel_list):

    # Read EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=False)

    # Get the list of all available channels in the file
    available_channels = raw.info['ch_names']

    # Find the intersection of the available channels and the EEG channel list
    eeg_channels_to_retain = [ch for ch in available_channels if ch in eeg_channel_list]

    # Retain only the identified EEG channels, if preload=True is used above
    # raw.pick(eeg_channels_to_retain)

    raw = mne.io.read_raw_edf(edf_file_path, preload=True, include=eeg_channels_to_retain)

    return raw, eeg_channels_to_retain


path_file = r'/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/'
path_file2 = r'/media/livia/Elements/public_sleep_data/stages/stages/original/'
clinics = glob2.glob(path_file + "*")

for clinic in clinics:
    print("Clinic ", clinic)
    data_path = os.path.join(path_file, clinic, 'usable')
    dir_subjs = sorted(glob2.glob(data_path + "/*.edf"))
    directory1 = "yasa_outputs/"
    path_par1 = os.path.join(path_file, clinic, directory1)
    directory2 = "mask_yasa_5sec_originalfs_0.3to35Hz/"
    path_mask = os.path.join(path_par1, directory2)
    mask_files = sorted(glob2.glob(path_mask + "/*.csv"))
    directory3 = "eeg_bandpowers_30sec/"
    path_save = os.path.join(path_par1, directory3)
    # os.mkdir(path_save)

    # extract all the subject IDs
    names = []
    for num_subj in range(len(dir_subjs)):
        m = re.search('usable/(.+?).edf', dir_subjs[num_subj])
        if m:
            name = m.group(1)
        # name = name.replace('/', '')
        names.append(name)
    names = np.array(names)

    # retained_eeg_channels = []
    # samp_freq = []
    for subject in range(len(dir_subjs)):
        print('Start processing', names[subject]+'.edf ...')
        raw_eeg, eeg_channels = retain_eeg_channels_from_list(dir_subjs[subject], eeg_channel_list)
        # eeg_channels.insert(0, names[subject])
        # retained_eeg_channels.append(np.array(eeg_channels))
        sf = raw_eeg.info['sfreq']
        # samp_freq.append(np.array((names[subject], sf)))
        if subject == 0:
            print("Sampling frequency:", raw_eeg.info['sfreq'])

        raw_eeg.filter(0.3, 35)
        # raw.resample(100)
        # data = raw_eeg.get_data(units="uV")
        data = raw_eeg.get_data()
        print("Original data shape:", data.shape)  # (12, 7631800)

        # Load and apply 5-sec masks
        mask = np.loadtxt(mask_files[subject], delimiter=',', skiprows=0, dtype=float)
        mask_up = yasa.hypno_upsample_to_data(hypno=mask, sf_hypno=(1 / 5), data=data, sf_data=sf)
        print(mask_up.size == data.shape[1])  # Does the hypnogram have the same number of samples as data?
        where_clean = (mask_up == 0)  # True if sample is clean / False if it belongs to an artifactual 5-sec
        data_clean = data[:, where_clean]
        print("Clean data shape:", data_clean.shape)

        # matplotlib.use('TkAgg')
        # plt.figure()
        # plt.plot(data_clean[2, 0:10000])
        # plt.show()

        # Segment all the channels into "window" seconds
        seg_window = 30
        _, data_clean_segmented = yasa.sliding_window(data_clean, sf, window=seg_window)
        print(f"Data shape after cleaning and {seg_window}-sec segmentation (epochs, channels, samples):",
              data_clean_segmented.shape)

        # PSD calculation
        # Method1: Welch
        # welch_window_duration = 4
        # win = int(welch_window_duration * sf)  # Window size set to "welch_window_duration" seconds for PSD calculation
        # freqs, psd = welch(data_clean_segmented, sf, nperseg=win, axis=-1)
        # print("PSD shape (epochs, channels, frequencies):", psd.shape)
        # Method2: Multitaper
        psd, freqs = psd_array_multitaper(data_clean_segmented, sf, adaptive=False, normalization='full', verbose=0)  # bandwidth=4
        print("PSD shape (epochs, channels, frequencies):", psd.shape)

        # bandpower_from_psd_ndarray: expected input shape (n_freqs) or (n_chan, n_freqs) or (n_chan, n_epochs, n_freqs)
        #                             output shape (n_bands, n_chan, n_epochs)
        bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 16, 'Sigma'), (16, 30, 'Beta'),
                 (30, 35, 'Gamma')]
        bp_rel = yasa.bandpower_from_psd_ndarray(psd.transpose(1, 0, 2), freqs, bands, relative=True)
        bp_abs = yasa.bandpower_from_psd_ndarray(psd.transpose(1, 0, 2), freqs, bands, relative=False)

        # matplotlib.use('TkAgg')
        # a = bp[:, 3, :]  # select one channel, a contains the power of the 6 bands for each epoch
        # plt.figure(figsize=(10, 6))
        # sns.heatmap(data, cmap='viridis', annot=False)  # You can change 'viridis' to any other colormap
        # plt.show()

        path_sbj = os.path.join(path_save, names[subject])
        os.mkdir(path_sbj)
        bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']
        for ind in range(len(bands)):
            relative = pd.DataFrame(bp_rel[ind, :, :])
            absolute = pd.DataFrame(bp_abs[ind, :, :])
            relative.insert(0, 'Channel', eeg_channels)
            absolute.insert(0, 'Channel', eeg_channels)
            relative.to_csv(os.path.join(path_sbj, bands[ind]+'_rel_power_all_eeg_channels_30sec.csv'), index=False)
            absolute.to_csv(os.path.join(path_sbj, bands[ind] + '_abs_power_all_eeg_channels_30sec.csv'), index=False)

    aa=0









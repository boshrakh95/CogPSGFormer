
# Process and save EEG spectrograms from the STAGES dataset (only C3-M2 channel for now)
# Segments the EEG signal into 30-sec windows (no overlap) and computes the spectrogram for each segment
# Chose 30-sec windows to match the raw EEG signal segmentation in "EEG_processing_raw_STAGES_v1.py"
# Sleep stages not considered (still do not have reliable hypnograms)

# 12 Dec 2024
# Boshra

# To do:
# For best performance, apply yasa.art_detect on pre-staged data and make sure to pass the hypnogram.
# Sleep stages have very different EEG signatures and the artifect rejection will be much more accurate when applied separately on each sleep stage.

# To do2:
# There is still significant artifact in the signals even after yasa.art_detect(), remove them by thresholding later

# To do3:
# Ran to process all 4 channels (C3-M2, C4-M1, F3-M2, F4-M1) but only saved C3-M2 for now. Save the rest as well.

import mne
import yasa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import spectrogram  # , stft
import seaborn as sns
import glob2
import os
import re
from scipy.signal import decimate
# from mne.time_frequency import psd_array_multitaper

matplotlib.use('Agg')

# All possible EEG channel names in the STAGES dataset
eeg_channel_list = ['C3M2', 'C4M1', 'C3', 'C4', 'M1', 'M2', 'F3M2', 'F4M1', 'F3', 'F4']


def retain_and_rereference_channels(edf_file_path, eeg_channel_list):
    # Read only the metadata (channel names) from the EDF file without loading data
    raw_info = mne.io.read_raw_edf(edf_file_path, preload=False, verbose=False)

    # Get the list of all available channels in the file
    available_channels = raw_info.info['ch_names']

    # Check if any of the key channels or already referenced channels are present
    if not any(ch in available_channels for ch in eeg_channel_list):
        return None

    # Filter to include only channels that are in the provided list
    eeg_channels_to_retain = [ch for ch in available_channels if ch in eeg_channel_list]

    # Now load the EDF file including only the retained EEG channels
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, include=eeg_channels_to_retain)
    sf = raw.info['sfreq']

    # Check if already referenced channels are present
    results = {}
    if 'C3M2' in available_channels:
        results['C3-M2'] = raw.copy().pick(['C3M2']).get_data()
    if 'C4M1' in available_channels:
        results['C4-M1'] = raw.copy().pick(['C4M1']).get_data()
    if 'F3M2' in available_channels:
        results['F3-M2'] = raw.copy().pick(['F3M2']).get_data()
    if 'F4M1' in available_channels:
        results['F4-M1'] = raw.copy().pick(['F4M1']).get_data()

    # If not all pre-referenced channels are available, check for separate channels
    if all(ch in eeg_channels_to_retain for ch in ['C3', 'C4', 'M1', 'M2']):
        if 'C3-M2' not in results:
            raw_c3_m2 = raw.copy().set_eeg_reference(ref_channels=['M2']).pick(['C3']).get_data()
            results['C3-M2'] = raw_c3_m2
        if 'C4-M1' not in results:
            raw_c4_m1 = raw.copy().set_eeg_reference(ref_channels=['M1']).pick(['C4']).get_data()
            results['C4-M1'] = raw_c4_m1

    if all(ch in eeg_channels_to_retain for ch in ['F3', 'F4', 'M1', 'M2']):
        if 'F3-M2' not in results:
            raw_f3_m1 = raw.copy().set_eeg_reference(ref_channels=['M2']).pick(['F3']).get_data()
            results['F3-M2'] = raw_f3_m1
        if 'F4-M1' not in results:
            raw_f4_m2 = raw.copy().set_eeg_reference(ref_channels=['M1']).pick(['F4']).get_data()
            results['F4-M1'] = raw_f4_m2

    return results, sf if results else None


def compute_fft_spectrogram(eeg_signal, fs, window_size, step_size, nfft):
    # Compute the spectrogram using a sliding window approach
    f, t, Sxx = spectrogram(eeg_signal, fs=fs, window='hamming', nperseg=window_size, noverlap=(window_size - step_size), nfft=nfft)
    # Optionally apply logarithmic scaling
    #Sxx_log = np.log1p(Sxx)
    return f, t, Sxx  #_log


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
    directory3 = "eeg_spectrogram_30sec/"
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

    # retained_eeg_channels = []
    # samp_freq = []
    for subject in range(len(dir_subjs)):
        # subject = subject + 32

        print('Start processing', names[subject]+'.edf ...')
        results = retain_and_rereference_channels(dir_subjs[subject], eeg_channel_list)

        if results:
            data_and_channels, sf = results
            print("Re-referenced signals found:")
            for key, value in data_and_channels.items():
                print(f"{key}: Shape = {value.shape}")

            # # eeg_channels.insert(0, names[subject])
            # # retained_eeg_channels.append(np.array(eeg_channels))
            # sf = raw_eeg.info['sfreq']
            # # samp_freq.append(np.array((names[subject], sf)))
            if subject == 0:
                print(f"Sampling frequency: {sf}")

            filtered_data = []
            for channel_name, channel_data in data_and_channels.items():
                print(f"Processing {channel_name}...")

                # Create an MNE RawArray for each re-referenced channel
                info = mne.create_info(ch_names=[channel_name], sfreq=sf, ch_types=['eeg'])
                raw_channel = mne.io.RawArray(channel_data, info)
                raw_channel.filter(0.3, 35)
                filtered_data.append(raw_channel.get_data())

            if len(filtered_data) > 0:
                data = np.vstack(filtered_data)
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
                # plt.show(block=False)

                # Downsample to 70 Hz
                sf_old = sf
                sf = 70
                decimation_factor = int(sf_old / sf)
                data_clean = decimate(data_clean, q=decimation_factor, axis=1, zero_phase=True)
                # plt.figure()
                # plt.plot(data_clean[2, 0:10000//decimation_factor])
                # plt.show(block=False)

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
                # psd, freqs = psd_array_multitaper(data_clean_segmented, sf, adaptive=False, normalization='full',
                #                                   verbose=0)  # bandwidth=4
                # print("PSD shape (epochs, channels, frequencies):", psd.shape)

                # bandpower_from_psd_ndarray: expected input shape (n_freqs) or (n_chan, n_freqs) or (n_chan, n_epochs, n_freqs)
                #                             output shape (n_bands, n_chan, n_epochs)
                # bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 16, 'Sigma'), (16, 30, 'Beta'),
                #          (30, 35, 'Gamma')]
                # bp_rel = yasa.bandpower_from_psd_ndarray(psd.transpose(1, 0, 2), freqs, bands, relative=True)
                # bp_abs = yasa.bandpower_from_psd_ndarray(psd.transpose(1, 0, 2), freqs, bands, relative=False)

                # matplotlib.use('TkAgg')
                # a = bp[:, 3, :]  # select one channel, a contains the power of the 6 bands for each epoch
                # plt.figure(figsize=(10, 6))
                # sns.heatmap(data, cmap='viridis', annot=False)  # You can change 'viridis' to any other colormap
                # plt.show()

                # STFT on each 1min data
                # fs = 64  # Sampling frequency (Hz)
                window_size = sf  # Length of each segment for STFT (nperseg)
                step_size = window_size   # * 3 / 4  # np.round(75*window_size/100)  # 25% overlap
                nfft = window_size
                # method1
                # f, t, Sxx = stft(data, fs=fs, nperseg=window_size, noverlap=window_size - step_size)
                # Zxx = np.log1p(np.abs(Sxx))
                #  method2
                # Extract channel C3-M2 --> modify this when adding more channels
                eeg_channels = list(data_and_channels.keys())
                # find the index of the C3-M2 channel
                ind = eeg_channels.index('C3-M2')
                data_clean_segmented = data_clean_segmented[:, ind, :]
                f, t, Zxx_eeg = compute_fft_spectrogram(eeg_signal=data_clean_segmented, fs=sf,
                                                        window_size=window_size, step_size=step_size, nfft=nfft)

                # matplotlib.use('TkAgg')
                # plt.figure(figsize=(10, 6))
                # plt.pcolormesh(t, f, Zxx_eeg[200, :, :], shading='gouraud')
                # plt.show()

                path_sbj = os.path.join(path_save, names[subject])
                os.makedirs(path_sbj, exist_ok=True)
                # bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']
                eeg_channels = ['C3-M2']  # list(data_and_channels.keys()) for now just use C3 channel
                # Save the segmented data of each channel in a numpy file
                for ind in range(len(eeg_channels)):
                    np.save(os.path.join(path_sbj, 'eeg_' + eeg_channels[ind] + '_spect_30sec.npy'),
                            Zxx_eeg)

                if subject == 0:
                    np.save(os.path.join(path_file2,
                                         'yasa_eeg_powers/eeg_spect_window70_step70_segmented_30sec_frequency.npy'), f)
                    np.save(
                        os.path.join(path_file2, 'yasa_eeg_powers/eeg_spect_window70_step70_segmented_30sec_time.npy'),
                        t)
        else:
            print("Required channels not found or not all present.")

    aa=0









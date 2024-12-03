
# RUN yasa artifact rejection on the STAGES dataset
# Sleep stages not considered (still do not have reliable hypnograms)
# 21 Oct 2024
# Boshra

# To do:
# For best performance, apply yasa.art_detect on pre-staged data and make sure to pass the hypnogram.
# Sleep stages have very different EEG signatures and the artifect rejection will be much more accurate when applied separately on each sleep stage.

import mne
import yasa
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
import glob2
import os
import re

# All possible EEG channel names in the STAGES dataset
eeg_channel_list = ['E1M2', 'E2M2', 'C3M2', 'C4M1', 'O1M2', 'O2M1',
                    'F3M2', 'F4M1', 'F1M2', 'F2M1', 'T3M2', 'T4M1', 'Pz', 'P4', 'EEG_F3_A2', 'EEG_F4_A1', 'EEG_A1_A2',
                    'EEG_C3_A2', 'EEG_C4_A1', 'EEG_O1_A2', 'EEG_O2_A1', 'EEG_T3_A2', 'EEG_T4_A1', 'EEG_P3_A2',
                    'EEG_P4_A1', 'Fpz', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'Oz', 'O1', 'O2', 'M1', 'M2',
                    'Fz', 'Cz', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'EEG_Fp1_A2', 'EEG_Fp2_A1', 'EEG_Fp1_A22',
                    'EEG_Fp2_A12', 'EEG_F3_A22', 'EEG_F4_A12', 'EEG_A1_A22', 'EEG_C3_A22', 'EEG_C4_A12', 'EEG_O1_A22',
                    'EEG_O2_A12', 'EEG_F3_A1', 'EEG_F4_A2', 'EEG_C3_A1', 'EEG_C4_A2',
                    'EEG_O1_A1', 'EEG_O2_A2', 'EEG_Fp1_A22', 'E1', 'E2', 'EEG_F3-A2', 'EEG_F4-A1',
                    'EEG_A1-A2', 'EEG_C3-A2', 'EEG_C4-A1', 'EEG_O1-A2', 'EEG_O2-A1',]


# Read the EDF file and retain all the EEG channels by searching through "eeg_channel_list"
def retain_eeg_channels_from_list(edf_file_path, eeg_channel_list):

    # Read EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=False)

    # Get the list of all available channels in the file
    available_channels = raw.info['ch_names']

    # Find the intersection of the available channels and the EEG channel list
    eeg_channels_to_retain = [ch for ch in available_channels if ch in eeg_channel_list]

    # Retain only the identified EEG channels
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
    os.mkdir(path_par1)
    directory2 = "mask_yasa_5sec_originalfs_0.3to35Hz/"
    path_par2 = os.path.join(path_par1, directory2)
    # if 'BOGN' not in clinic:
    os.mkdir(path_par2)

    # extract all the subject IDs
    names = []
    for num_subj in range(len(dir_subjs)):
        m = re.search('usable/(.+?).edf', dir_subjs[num_subj])
        if m:
            name = m.group(1)
        # name = name.replace('/', '')
        names.append(name)
    names = np.array(names)

    retained_eeg_channels = []
    samp_freq = []
    for subject in range(len(dir_subjs)):
        print('Start processing', names[subject]+'.edf ...')
        raw_eeg, eeg_channels = retain_eeg_channels_from_list(dir_subjs[subject], eeg_channel_list)
        eeg_channels.insert(0, names[subject])
        retained_eeg_channels.append(np.array(eeg_channels))
        sf = raw_eeg.info['sfreq']
        samp_freq.append(np.array((names[subject], sf)))
        if subject == 0:
            print("Sampling frequency:", raw_eeg.info['sfreq'])

        raw_eeg.filter(0.3, 35)
        # raw.resample(100)
        data = raw_eeg.get_data(units="uV")
        print("Data shape:", data.shape)  # (12, 7631800)

        art, zscores = yasa.art_detect(data, sf, window=5, method='covar', threshold=3)
        print("Mask shape:", art.shape)  # (7631,)
        # Art is an aray of 0 and 1, where 0 indicates a clean (or good epoch)  and 1 indicates an artifact epoch
        print(f'{art.sum()} / {art.size} epochs rejected.')  # for BOGN0004: 38 / 7631 epochs rejected.

        # If you want the mask vector to have the same shape as the signal (upsample)
        # The resolution of art is 5 seconds, so its sampling frequency is 1/5 (= 0.2 Hz)
        # sf_art = 1 / 5
        # art_up = yasa.hypno_upsample_to_data(art, sf_art, data, sf)

        np.savetxt(os.path.join(path_par2, names[subject]+'_mask_5sec.csv'), art, delimiter=',', fmt='%d')
        # a=1

    df1 = pd.DataFrame(retained_eeg_channels)
    df1.to_csv(path_par1+'eeg_channels_used_for_yasa_artifact_detection.csv', index=False, header=False)
    df2 = pd.DataFrame(samp_freq)
    df2.to_csv(path_par1+'eeg_sampling_frequencies_in_STAGES.csv', index=False, header=False)










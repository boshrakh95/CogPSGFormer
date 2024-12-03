
# RUN yasa artifact rejection on a sample edf from the STAGES dataset
# compare yasa and luna masks visually to decide which algorithm to use (results: use yasa)
# 16 Oct 2024
# Boshra

import mne
import yasa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib

matplotlib.use('Agg')

raw = mne.io.read_raw_edf('/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/BOGN/usable/BOGN00008.edf', preload=True)
# raw = mne.io.read_raw_edf('/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/MAYO/usable/MAYO00046.edf', preload=True)
# raw = mne.io.read_raw_edf('/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/STNF/usable/STNF00005.edf', preload=True)
# /media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/GSBB/usable

print("all channels:", raw.ch_names)
# BOGN:
# all channels: ['E1M2', 'E2M2', 'C3M2', 'C4M1', 'O1M2', 'O2M1', 'F3M2', 'F4M1', 'F1M2', 'F2M1', 'FLOW', 'PTAF',
#                'SNOR', 'EKG', 'THOR', 'ABDM', 'RLEG', 'LLEG', '32', 'CFLO', 'LEAK', 'IPAP', 'EPAP', 'ETC2',
#                'WAVE', 'POS', 'DC08', 'T3M2', 'T4M1', 'SpO2', 'Plth', 'CHIN', 'Pz', 'P4']

# raw.drop_channels(['FLOW', 'PTAF', 'SNOR', 'EKG', 'THOR', 'ABDM', 'RLEG', 'LLEG', '32', 'CFLO', 'LEAK', 'IPAP', 'EPAP', 'ETC2', 'WAVE', 'POS', 'DC08', 'T3M2', 'T4M1', 'SpO2', 'Plth', 'CHIN'])
raw.drop_channels(['FLOW', 'PTAF', 'SNOR', 'EKG', 'THOR', 'ABDM', 'RLEG', 'LLEG', 'CFLO', 'LEAK', 'IPAP', 'EPAP', 'ETC2', 'WAVE', 'POS', 'DC08', 'SpO2', 'Plth', 'CHIN'])
chan = raw.ch_names
print("remaining channels:", chan)
# remaining channels: ['E1M2', 'E2M2', 'C3M2', 'C4M1', 'O1M2', 'O2M1', 'F3M2', 'F4M1', 'F1M2', 'F2M1', 'Pz', 'P4']

print(raw.info['sfreq'])  # 200 Hz

# raw.resample(100)
sf = raw.info['sfreq']
# print("New sampling frequency:", sf)  # 100 Hz

# Band-pass filter the signals
raw.filter(0.3, 45)
# raw.filter(0.3, 30)
# raw.resample(64)
# sf = 64

data = raw.get_data(units="uV")
print(data.shape)  # (12, 7631800)

art, zscores = yasa.art_detect(data, sf, window=5, method='covar', threshold=3)
print(art.shape)  # (7631,)

# Art is an aray of 0 and 1, where 0 indicates a clean (or good epoch)  and 1 indicates an artifact epoch
print(art)
print(f'{art.sum()} / {art.size} epochs rejected.')  # for BOGN0004: 38 / 7631 epochs rejected.

# Plot the artifact vector
# plt.plot(art)
# plt.yticks([0, 1], labels=['Good (0)', 'Art (1)'])

# The resolution of art is 5 seconds, so its sampling frequency is 1/5 (= 0.2 Hz)
sf_art = 1 / 5
art_up = yasa.hypno_upsample_to_data(art, sf_art, data, sf)
print(art_up.shape)

data1 = data[6, :]
del data, raw

matplotlib.use('TkAgg')
plt.figure()
plt.plot(data1)
plt.plot(art_up*max(data1)/2)
plt.show()




###################################################################################################################
# Load and plot luna processed data and artifact mask
# to compare luna and yasa masks visually and decide which artifact detection to use

raw1 = mne.io.read_raw_edf('/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/BOGN/processed_luna/BOGN00012-luna_processed.edf', preload=True)
print("all channels:", raw1.ch_names)

raw1.drop_channels(['EKG'])
chan = raw1.ch_names
print("remaining channels:", chan)
# remaining channels: ['F3M2', 'F4M1']

print(raw1.info['sfreq'])  # 64.0 Hz
sf1 = raw1.info['sfreq']

data2 = raw1.get_data(units="uV")
print(data2.shape)  # (2, 2442176)
data2 = data2[0, :]

luna_mask = pd.read_csv("/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/BOGN/mask_luna/mask_BOGN00012_EEG.csv")
luna_mask = luna_mask.values.ravel()

# method1 for upsampling the mask to data and zeropad (from the end)
# luna_mask_up = yasa.hypno_upsample_to_data(luna_mask, 1/30, data2, sf1)

# method2 for upsampling the mask to data and zeropad (from the end or beginning, (0, pad_length) or (pad_length, 0))
luna_mask_up = np.repeat(luna_mask, 30*sf1)
pad_length = len(data2) - len(luna_mask_up)  # Compute the difference in length
if pad_length > 0:  # Pad arr1 with zeros to match the length of arr2
    luna_mask_up = np.pad(luna_mask_up, (0, pad_length), mode='constant', constant_values=0)
else:
    luna_mask_up = luna_mask_up

plt.figure()
plt.plot(data2)
plt.plot(luna_mask_up*max(data2)/2)

# Conclusion:
# upsampled luna masks are 59sec shorter than the signals (for all subjects), padding from neither start
# or end gives synchronization. It's something in between. Not sure how to sync it.
# So prefer to use yasa masks over luna masks. Also yasa results make a bit more sense visually and the detection
# resolution can be better than luna (here I used 5sec windows, while for luna it's always 30sec)






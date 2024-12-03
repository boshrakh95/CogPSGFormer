
# RUN yasa on a sample edf from the STAGES dataset
# 3 Oct 2024
# Boshra

import mne
import yasa
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re

raw = mne.io.read_raw_edf('/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/BOGN/usable/BOGN00004.edf', preload=True)
# raw = mne.io.read_raw_edf('/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/MAYO/usable/MAYO00046.edf', preload=True)
# raw = mne.io.read_raw_edf('/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/STNF/usable/STNF00005.edf', preload=True)
# /media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/GSBB/usable

print("all channels:", raw.ch_names)
#['ROC-A1', 'LOC-A2', 'C3-A2', 'O2-A1', 'C4-A1', 'O1-A2', 'EMG1-EMG2', 'Fp1-A2', 'Fp2-A1', 'F7-A2',
 #'F3-A2', 'FZ-A2', 'F4-A1', 'F8-A1', 'T3-A2', 'CZ-A2', 'T4-A1', 'T5-A2', 'P3-A2', 'PZ-A2', 'P4-A1',
 #'T6-A1', 'EKG-R-EKG-L']

# raw.drop_channels(['ROC-A1', 'LOC-A2', 'EMG1-EMG2', 'EKG-R-EKG-L'])
chan = raw.ch_names
print("remaining channels:", chan)
#['C3-A2', 'O2-A1', 'C4-A1', 'O1-A2', 'Fp1-A2', 'Fp2-A1', 'F7-A2', 'F3-A2', 'FZ-A2',
 #'F4-A1', 'F8-A1', 'T3-A2', 'CZ-A2', 'T4-A1', 'T5-A2', 'P3-A2', 'PZ-A2', 'P4-A1', 'T6-A1']

print(raw.info['sfreq'])  # 200 Hz

# raw.resample(100)
sf = raw.info['sfreq']
# print("New sampling frequency:", sf)  # 100 Hz
#
# raw.filter(0.3, 45)

from visbrain.gui import Sleep
Sleep().show()

data = raw.get_data(units="uV")
print(data.shape)  # (19, 2892000)

# hypno = pd.read_csv("/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/BOGN/usable/BOGN00004.csv")
hypno = pd.read_csv("/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/MAYO/usable/MAYO00046.csv")
hypno = hypno['Event']  # only retain the third column
# hypno = hypno.drop(hypno.columns[0], axis=1)  # remove the first column
# print(hypno)
# print(type(hypno))

# Define the mapping for the five stages
stage_mapping = {
    ' Wake': 0,
    ' Stage1': 1,
    ' Stage2': 2,
    ' Stage3': 3,
    ' REM': 4
}

# Apply the mapping and convert other stages to NaN
hypno_mapped = hypno.map(stage_mapping)

# Drop any rows with NaN values (i.e., those stages you don't want to retain)
hypno_mapped = hypno_mapped.dropna()

# Convert to integers (optional, if you need it in int form)
hypno = hypno_mapped.astype(int)

# # Define a mapping for the five stages, using regex patterns
# stage_mapping = {
#     r'(?i)wake': 0,      # '(?i)' makes the match case-insensitive
#     r'(?i)stage1': 1,
#     r'(?i)stage2': 2,
#     r'(?i)stage3': 3,
#     r'(?i)rem': 4
# }
#
# # Function to map the stages using regex
# def map_stages(value):
#     for pattern, code in stage_mapping.items():
#         if re.search(pattern, value):  # Search for the pattern in the string
#             return code
#     return None  # Return None if no pattern matches
#
# # Apply the mapping function to the stage column
# hypno['stage_mapped'] = hypno['Event'].apply(map_stages)
#
# # Replace NaN values with 6
# hypno['stage_mapped'] = hypno['stage_mapped'].fillna(6)
#
# # Repeat the rows based on 'repeat_col' value
# repeated_stages = hypno.apply(lambda row: [row['stage_mapped']] * (row['Duration (seconds)'] if row['Duration (seconds)'] > 0 else 1), axis=1)
#
# # Flatten the list of lists into a single list
# repeated_stages_flat = [item for sublist in repeated_stages for item in sublist]
#
# # Convert the list back to a Pandas Series (or DataFrame if you prefer)
# repeated_stages_series = pd.Series(repeated_stages_flat)
#
# # Convert to integers
# hypno = hypno_mapped.astype(int)

yasa.plot_hypnogram(hypno)

print(yasa.sleep_statistics(hypno, sf_hyp=1/30))

# Sleep stages transition matrix
counts, probs = yasa.transition_matrix(hypno)
print(probs.round(3))

# Stability of sleep: average of the diagonal values of N2, N3 and REM sleep
np.diag(probs.loc[2:, 2:]).mean().round(3)  # 0.867

# Spectral analysis

# Full-night spectrogram plot
hypno_up = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw)
print(len(hypno_up))
yasa.plot_spectrogram(data[chan.index("C4")], sf, hypno_up)  # We select only the C4-A1 EEG channel.

# EEG power in specific frequency bands
yasa.bandpower(raw)

yasa.bandpower(raw, relative=False, bands=[(1, 9, "Slow"), (9, 30, "Fast")])

bandpower = yasa.bandpower(raw, hypno=hypno_up, include=(2, 3, 4))
# bandpower.to_csv("bandpower.csv")

fig = yasa.topoplot(bandpower.xs(3)['Delta'])  # topography of Delta power in stage N3

# Event detection

# Spindles
sp = yasa.spindles_detect(raw, hypno=hypno_up, include=(2, 3))
sp.summary()
sp.summary(grp_chan=True, grp_stage=True)

# Plot the average spindle, calculated for each channel separately and time-synced to the most prominent spindle peak
# Because of the large number of channels, we disable the 95%CI and legend
sp.plot_average(errorbar=None, legend=False, palette="Blues")

# Slow waves
sw = yasa.sw_detect(raw, hypno=hypno_up, include=(2, 3))
sw.summary()

sw.plot_average(errorbar=None, legend=False, palette="Blues")

# Automatic sleep staging
sls = yasa.SleepStaging(raw, eeg_name='C3-A2')
hypno_pred = sls.predict()  # Predict the sleep stages
hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc

yasa.plot_hypnogram(hypno_pred)
plt.show()

print(f"The accuracy is {100 * accuracy_score(hypno, hypno_pred):.3f}%")  # The accuracy is 82.676%


a = 1
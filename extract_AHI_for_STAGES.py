import pandas as pd
import numpy as np
import re

# Load the Excel file into a DataFrame
file_path = '/media/livia/Elements/public_sleep_data/stages/stages/original/De-identified Data/' \
            'PSG SRBD Variables/STAGESPSGKeySRBDVariables2020-08-29 Deidentified.xlsx'
df = pd.read_excel(file_path)

# List of individuals in the subset
path_file2 = r'/media/livia/Elements/public_sleep_data/stages/stages/original/'
subj_retained_for_power_analysis = pd.read_csv(path_file2 + "yasa_eeg_powers/subjects_retained_for_power_analysis.csv", header=None)
subj_retained_for_power_analysis = subj_retained_for_power_analysis.values.flatten().tolist()
names_ret = []
for num_subj in range(len(subj_retained_for_power_analysis)):
    m = re.search('.*/([^/]+)$', subj_retained_for_power_analysis[num_subj])
    if m:
        name = m.group(1)
    # name = name.replace('/', '')
    names_ret.append(name)
names_ret = np.array(names_ret)

missing_ids = [name for name in names_ret if name not in df['s_code'].values]
print("Missing IDs:", missing_ids)
# Missing IDs: ['MAYO00029', 'MAYO00030', 'MAYO00059', 'MAYO00064', 'MSQW00017',
# 'MSTH00019', 'MSTH00023', 'MSTR00096', 'MSTR00097', 'MSTR00204', 'STNF00023',
# 'STNF00029', 'STNF00067', 'STNF00080', 'STNF00095', 'STNF00121', 'STNF00130',
# 'STNF00135', 'STNF00168', 'STNF00187', 'STNF00199_1', 'STNF00237', 'STNF00240',
# 'STNF00241', 'STNF00256', 'STNF00300', 'STNF00304', 'STNF00309', 'STNF00317',
# 'STNF00325', 'STNF00332', 'STNF00337', 'STNF00339', 'STNF00346', 'STNF00361',
# 'STNF00364', 'STNF00368', 'STNF00376', 'STNF00398', 'STNF00409', 'STNF00415',
# 'STNF00425', 'STNF00453', 'STNF00461', 'STNF00477', 'STNF00486', 'STNF00488']

missing_indexes = np.where(np.isin(names_ret, missing_ids))[0]

df['s_code'] = df['s_code'].str.strip().str.upper()  # Standardize formatting
names_ret = [name.strip().upper() for name in names_ret]  # Standardize the subset list

available_ids = [name for name in names_ret if name in df['s_code'].values]
ahi_array = df.loc[df['s_code'].isin(available_ids)].set_index('s_code').loc[available_ids, 'ahi'].values

print(len(names_ret) - len(available_ids), "IDs were not found in the DataFrame.")



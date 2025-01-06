import pandas as pd

# Load the patient demographics Excel file
demographics_file = '/media/livia/Elements/public_sleep_data/stages/stages/original/De-identified Data/PSG SRBD Variables/STAGESPSGKeySRBDVariables2020-08-29 Deidentified.xlsx'
demographics_df = pd.read_excel(demographics_file)

# Retain only relevant columns
demographics_df = demographics_df[['s_code', 'age', 'sex', 'bmi', 'sleep_time']]

# Load the CSV file containing IDs for analysis
ids_file = '/media/livia/Elements/public_sleep_data/stages/stages/original/yasa_eeg_powers/final_subjs_pcet_for_demog.csv'
ids_df = pd.read_csv(ids_file, header=None, names=['s_code'])

# Extract IDs for analysis
ids_to_analyze = ids_df['s_code'].unique()

# Filter demographics data to retain rows with IDs in the analysis list
filtered_demographics_df = demographics_df[demographics_df['s_code'].isin(ids_to_analyze)]

# Find IDs in the analysis list that are not in the demographics file
missing_ids = [id_ for id_ in ids_to_analyze if id_ not in demographics_df['s_code'].values]

# Calculate min and max sleep duration for the filtered data
min_sleep_duration = filtered_demographics_df['sleep_time'].min()
max_sleep_duration = filtered_demographics_df['sleep_time'].max()

# Count of males and females
gender_counts = filtered_demographics_df['sex'].value_counts()

# Mean and standard deviation for BMI and age
mean_bmi = filtered_demographics_df['bmi'].mean()
std_bmi = filtered_demographics_df['bmi'].std()
mean_age = filtered_demographics_df['age'].mean()
std_age = filtered_demographics_df['age'].std()

# Display the results
# print("\nMissing IDs (no demographics available):")
# print(missing_ids)
print("\nGender Counts:")
print(gender_counts)
print(f"\nMean BMI: {mean_bmi:.2f}, Standard Deviation BMI: {std_bmi:.2f}")
print(f"Mean Age: {mean_age:.2f}, Standard Deviation Age: {std_age:.2f}")
print(f"\nMinimum Sleep Duration: {min_sleep_duration/3600}")
print(f"Maximum Sleep Duration: {max_sleep_duration/3600}")
a = 0
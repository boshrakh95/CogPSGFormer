import os, glob2

dir1 = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs/"

path =sorted(glob2.glob(dir1 + "*/yasa_outputs/ecg_hrv_params/*/*"))

# if found a file with the name "time_hrv_params_2min_50%overlap_2.v" in the directories in path, then
# change the name of the file to "time_hrv_params_2min_50%overlap_2.csv"
for file in path:
    if "time_hrv_params_2min_50%overlap_2.v" in file:
        os.rename(file, file.replace("time_hrv_params_2min_50%overlap_2.v", "time_hrv_params_2min_50%overlap_2.csv"))
    # else:
        # print("No file found with the name 'time_hrv_params_2min_50%overlap_2.v' in the directories in path.")
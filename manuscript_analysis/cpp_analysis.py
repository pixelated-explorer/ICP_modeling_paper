# %%

import pandas as pd
import numpy as np 
import missingno as mno
import seaborn as sns 
import matplotlib.pyplot as plt
'''
1. copy paste code from cleaning_graphs.py to generate cleaner vitalsP output 
2. use algoriithm below to stitch and create cpp values 

iter through FINAL__wce.... .csv 
    find patient id, find hours associated with patient id.
    -> match same patient id and same hour range with the vitalsP csv.
    add new column in FINAL being corresponding systemic_mean - cpp. 
'''

df_final = pd.read_csv(r"C:\ICP_analysis_paper\ICP_modeling_paper\Older files\result_files\FINAL_wce_merged_model.csv")
df_vitals = pd.read_csv(r"/home/rayan/Research/medical_proj/ICP_modeling_paper/Older files/older_graphing/vitalsP_cleaned.csv")
df_vitals = df_vitals.sort_values(by=['patientunitstayid', 'Time']) 

print(len(df_final['patientunitstayid'].unique()))

print(df_vitals.shape)

# %%

# Get unique patient IDs from both DataFrames
patient_ids_final = set(df_final['patientunitstayid'].unique())
patient_ids_vitals = set(df_vitals['patientunitstayid'].unique())

# Find patient IDs present in df_final but not in df_vitals
patient_ids_in_final_not_in_vitals = patient_ids_final - patient_ids_vitals
print(f"Patient IDs in df_final but not in df_vitals: {patient_ids_in_final_not_in_vitals}")

# Find patient IDs present in df_vitals but not in df_final
patient_ids_in_vitals_not_in_final = patient_ids_vitals - patient_ids_final
print(f"Patient IDs in df_vitals but not in df_final: {patient_ids_in_vitals_not_in_final}")

# We will now filter by the patients in df_final for graphing purposes 

df_vitals = df_vitals[(df_vitals['patientunitstayid'].isin(df_final['patientunitstayid']))]
print(df_vitals.shape)

# %%

# NOTE: double checking that diff sets are both empty

patient_ids_final = set(df_final['patientunitstayid'].unique())
patient_ids_vitals = set(df_vitals['patientunitstayid'].unique())


patient_ids_in_final_not_in_vitals = patient_ids_final - patient_ids_vitals
print(f"Patient IDs in df_final but not in df_vitals: {patient_ids_in_final_not_in_vitals}")

patient_ids_in_vitals_not_in_final = patient_ids_vitals - patient_ids_final
print(f"Patient IDs in df_vitals but not in df_final: {patient_ids_in_vitals_not_in_final}")

# %%

print(mno.matrix(df_final, figsize = (20,6)))
print(mno.matrix(df_vitals, figsize = (20,6)))

# %%

df_vitals.head()

# %%

df_final.head()


# %%
'''
NOTE: This code block filters by the hours that are listed in final wce merged. does NOT aggregate.

# cpp_list = []

# for i, patient in enumerate(df_final['patientunitstayid'].unique()):
#     final_slice = df_final[df_final['patientunitstayid'] == patient]
#     vitals_slice = df_vitals[
#         (df_vitals['patientunitstayid'] == patient) & 
#         (df_vitals['Hour'].isin(final_slice['Hour']))
#     ]
    
#     temp_cpp_list = vitals_slice['systemicmean'] - vitals_slice['icp']
#     cpp_list.extend(temp_cpp_list.values)

# print(len(cpp_list))
'''

# %%

# # review cpp list after extending each patient cpp portion to the main list 
# print(cpp_list)

# # total list of all cpp values, good
# print(len(cpp_list))
# length of original final dataframe, good
print(len(df_final))
# rows where there is a systemicmean 
print(df_vitals['systemicmean'].notna().sum())


# %%

# take original csv
# change nans in systemicmean to 999999
# make a column cpp which is difference between systemicmean and icp
# cut patients after

# Create a new column 'cpp' which is the difference between 'systemicmean' and 'icp'
df_vitals['cpp'] = df_vitals['systemicmean'] - df_vitals['icp']

print(df_vitals['cpp'])

# %%

plt.figure(figsize=(10,6))

# NOTE: PLOT CODE FOR CHECKING, CORR SHOWS THE SAME IDEA ANYWAY IN CELL BELOW 

# for patient, df_patient in df_vitals.groupby('patientunitstayid'):
    # sns.scatterplot(x=df_patient['icp'], y=df_patient['cpp'])
    # plt.show()
# %%
corr_value = df_vitals['cpp'].corr(df_vitals['icp'])

print(f"Correlation between CPP and ICP: {corr_value:.2f}")
# %%

# import matplotlib.pyplot as plt
# import seaborn as sns 
# from scipy.signal import savgol_filter

# # list of expired patient id's
# expiredID_list = [306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
#     2887235, 2898120, 3075566, 3336597]
# # list of alive patient id's
# aliveID_list = [193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
#     1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
#     2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
#     2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
#     2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
#     3217832, 3222024, 3245093, 3347750, 2782239]

# plt.figure(figsize=(13, 13))

# for patient in aliveID_list:
#     final_slice = df_vitals[df_vitals['patientunitstayid'] == patient]    
#     sns.lineplot(x=final_slice['Time'], y=final_slice['cpp'], label=f"{patient} patient")
    
# # Set the title and show the plot
# plt.title("Alive Patients' CPP Over Time")
# plt.xlabel("Time")
# plt.ylabel("CPP")
# plt.legend().set_visible(False)
# plt.savefig("alive_patients_cpp.png", dpi=600)
# plt.show()


# plt.figure(figsize=(13, 13))


# for patient in expiredID_list:
#     final_slice = df_vitals[df_vitals['patientunitstayid'] == patient]    
#     sns.lineplot(x=final_slice['Time'], y=final_slice['cpp'], label=f"{patient} patient")
    
# # Set the title and show the plot
# plt.title("Expired Patients' CPP Over Time")
# plt.xlabel("Time")
# plt.ylabel("CPP")
# plt.legend().set_visible(False)
# plt.savefig("expired_patients_cpp.png", dpi=600)
# plt.show()


# %%

df_vitals.head()


# %%
# NOTE: convert to medians, every 5 minutes 
'''
expiredID_list = [306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597]
# list of alive patient id's
aliveID_list = [193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239]


df_expired_med = df_vitals[df_vitals['patientunitstayid'].isin(expiredID_list)]
df_alive_med = df_vitals[df_vitals['patientunitstayid'].isin(aliveID_list)]

# NOTE: aggregate NOT for each patient, but for that time series for all patients. then see error bars. 


def med_func(df):
    df['time_minutes'] = df['Time'] * 60

    bin_edges = np.arange(0, df['time_minutes'].max() + 5, 5)  # Bins from 0 to max time with step of 5

    # Assign each row to a bin
    df['time_bin'] = pd.cut(df['time_minutes'], bins=bin_edges, right=False, labels=bin_edges[:-1])

    # Calculate median cpp for each 5-minute bin, grouped by patient
    df_median = df.groupby(['patientunitstayid', 'time_bin'])['cpp'].median().reset_index()

    return df_median

df_expired_med2 = med_func(df_expired_med)
df_alive_med2 = med_func(df_alive_med)
'''
# %%
'''
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.signal import savgol_filter


plt.figure(figsize=(13, 13))

sns.lineplot(x=df_expired_med['time_bin'], y=df_expired_med['cpp'])

plt.figure(figsize=(13, 13))


# for patient in expiredID_list:  
#     sns.lineplot(x=final_slice['time_bin'], y=final_slice['cpp'], label=f"{patient} patient")
    
# # Set the title and show the plot
# plt.title("Expired Patients' CPP Over Time")
# plt.xlabel("Time")
# plt.ylabel("CPP")
# plt.legend().set_visible(False)
# plt.savefig("expired_patients_cpp.png", dpi=600)
# plt.show()
'''

# %%

# NOTE: Now every 24 hours. 
# Segment the dataset based on `comp_outcome`

# Lists of expired and alive patients
expiredID_list = [306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597]

aliveID_list = [193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239]


expired_data = df_vitals[df_vitals['patientunitstayid'].isin(expiredID_list)]
alive_data = df_vitals[df_vitals['patientunitstayid'].isin(aliveID_list)]

# Function to calculate median and error bars for AUCHr every 24 hours
def calculate_median_and_error(data, time_column='Time', value_column='cpp', interval=24):
    # Bin the hours into intervals of `interval` (e.g., 24 hours)
    data['HourBin'] = (data[time_column] // interval) * interval

    # Group by the binned hours and calculate the median, 25th, and 75th percentiles
    grouped = data.groupby('HourBin')[value_column].agg(['median'])
    grouped['q25'] = data.groupby('HourBin')[value_column].quantile(0.25)
    grouped['q75'] = data.groupby('HourBin')[value_column].quantile(0.75)
    
    # Extract time (bin center), median, and error ranges
    time = grouped.index.values
    median = grouped['median'].values
    lower_error = median - grouped['q25'].values
    upper_error = grouped['q75'].values - median
    
    return time, median, lower_error, upper_error

# Calculate for both segments
time_alive, median_alive, lower_alive, upper_alive = calculate_median_and_error(alive_data)
time_expired, median_expired, lower_expired, upper_expired = calculate_median_and_error(expired_data)

# Plot both segments
plt.figure(figsize=(10, 6))

# Plot for "alive"
plt.errorbar(time_alive, median_alive, yerr=[lower_alive, upper_alive], fmt='-o', label='Alive', capsize=5)

# Plot for "expired"
plt.errorbar(time_expired, median_expired, yerr=[lower_expired, upper_expired], fmt='-o', label='Expired', capsize=5, color='red')

# Customize the plot
plt.title("Median CPP per 24-Hour Interval")
plt.xlabel("Time (hours)")
plt.ylabel("CPP")
plt.legend()
plt.grid(True)
plt.xticks(time_alive, rotation=45)  # Ensure x-axis ticks match 24-hour bins
plt.xlim(0, 192)  # Adjust as necessary based on your dataset

# Save and display the plot
# plt.savefig("median_Sp20AUCCum_24hr_plot.png", dpi=600)
# plt.show()

# %%

import numpy as np
import pandas as pd

# Lists of expired and alive patients
expiredID_list = [306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597]

aliveID_list = [193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239]

# Function to process and bin the time series data
def med_func(df):
    df['time_minutes'] = df['Time'] * 60  # Convert time to minutes

    # Create 5-minute bins
    bin_edges = np.arange(0, df['time_minutes'].max() + 5, 5)
    df['time_bin'] = pd.cut(df['time_minutes'], bins=bin_edges, right=False, labels=bin_edges[:-1])

    # Calculate per-patient median within each time bin
    df_median = df.groupby(['patientunitstayid', 'time_bin'])['cpp'].median().reset_index()
    return df_median

# Function to compute final aggregated values across all patients
def aggregate_medians(df_median):
    df_median['time_bin'] = df_median['time_bin'].astype(float)  # Ensure numeric sorting

    aggregated_df = df_median.groupby('time_bin')['cpp'].agg([
        ('median_cpp', 'median'),
        ('q25_cpp', lambda x: np.percentile(x, 25)),  # 25th percentile (Lower bound of IQR)
        ('q75_cpp', lambda x: np.percentile(x, 75)),  # 75th percentile (Upper bound of IQR)
        ('std_cpp', 'std'),  # Standard deviation
        ('sem_cpp', lambda x: x.std() / np.sqrt(len(x)))  # Standard error of mean
    ]).reset_index()

    return aggregated_df

# Filter and process expired patients
df_expired_med = df_vitals[df_vitals['patientunitstayid'].isin(expiredID_list)]
df_expired_med2 = med_func(df_expired_med)
df_expired_agg = aggregate_medians(df_expired_med2)

# Filter and process alive patients
df_alive_med = df_vitals[df_vitals['patientunitstayid'].isin(aliveID_list)]
df_alive_med2 = med_func(df_alive_med)
df_alive_agg = aggregate_medians(df_alive_med2)

# Print or return final dataframes
print(df_expired_agg.head())
print(df_alive_agg.head())


# %%

import matplotlib.pyplot as plt

# Function to plot with error bars
def plot_median_with_errors(df, title, color):
    # plt.figure(figsize=(15, 6))

    # Plot median line
    plt.plot(df['time_bin'], df['median_cpp'], label='Median CPP', color=color, linewidth=2)

    # Fill between Q25 and Q75 (IQR)
    plt.fill_between(df['time_bin'], df['q25_cpp'], df['q75_cpp'], color=color, alpha=0.3, label='IQR (25-75%)')

    # Error bars using standard error
    # plt.errorbar(df['time_bin'], df['median_cpp'], yerr=df['sem_cpp'], fmt='o', color=color, capsize=3, label='SEM')

    # Formatting
    plt.xlabel("Time (minutes)")
    plt.ylabel("CPP (Median ± Error)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.show()

plt.figure(figsize=(15, 6))

# Plot expired patients
plot_median_with_errors(df_expired_agg, "Expired Patients - CPP Over Time", "red")

# Plot alive patients
plot_median_with_errors(df_alive_agg, "Alive Patients - CPP Over Time", "blue")

plt.show()

# %%

import matplotlib.pyplot as plt
import numpy as np

# Function to plot segments
def plot_segmented(df, title, color, segment_length=1000):
    time_bins = df['time_bin'].values  # Extract time points
    max_time = int(time_bins.max())  # Find the max time for segmentation

    # Loop through segments
    for start in range(0, max_time, segment_length):
        end = start + segment_length
        df_segment = df[(df['time_bin'] >= start) & (df['time_bin'] < end)]

        if df_segment.empty:
            continue  # Skip if no data in this segment

        plt.figure(figsize=(10, 5))
        
        # Plot median line
        plt.plot(df_segment['time_bin'], df_segment['median_cpp'], label='Median CPP', color=color, linewidth=2)
        
        # Fill between Q25 and Q75 (IQR)
        plt.fill_between(df_segment['time_bin'], df_segment['q25_cpp'], df_segment['q75_cpp'], 
                         color=color, alpha=0.3, label='IQR (25-75%)')
        
        # Error bars using SEM
        plt.errorbar(df_segment['time_bin'], df_segment['median_cpp'], 
                     yerr=df_segment['sem_cpp'], fmt='o', color=color, capsize=3, label='SEM')
        
        # Formatting
        plt.xlabel("Time (minutes)")
        plt.ylabel("CPP (Median ± Error)")
        plt.title(f"{title} (Time {start}-{end} min)")
        plt.legend()
        plt.grid(True)
        plt.show()

# Plot expired patients in segments
plot_segmented(df_expired_agg, "Expired Patients - CPP Over Time", "red", segment_length=60)

# Plot alive patients in segments
plot_segmented(df_alive_agg, "Alive Patients - CPP Over Time", "blue", segment_length=60)

# %%


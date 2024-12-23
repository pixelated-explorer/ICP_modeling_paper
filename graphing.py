# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

# NOTE: CHANGE FROM HARDCODED FULL FILEPATH 
df = pd.read_csv(r"C:\ICP_analysis_paper\ICP_modeling_paper\Older files\result_files\FINAL_wce_merged_model.csv")

# %%
# find each patient id using .unique()
# save information into array of graphable information 
    # x axis time range, y axis the AUC cumulative 
# iterate through array, and plot each onto a matplotlib interface 
# at the end, do a .show(). render with 600 dpi 

unique_patients = df['patientunitstayid'].unique()

graph_data = []

for patient_id in unique_patients: 
    # Filter data for each patient
    patient_data = df[df['patientunitstayid'] == patient_id]
    graph_data.append((patient_data['Hour'].values, patient_data['AUCCum'].values))

plt.figure(figsize=(10,6))

# for loop for plot 

for time_data, auc_data in graph_data:
    plt.plot(time_data, auc_data, label=f'Patient {time_data[0]}')

plt.title("Cumulative AUC over Time for Each Patient")
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative AUC")
plt.xlim(24, 168)
# plt.yscale("log")
# plt.legend()


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Define the inset axes

ax_inset = inset_axes(plt.gca(), width="50%", height="30%", loc="upper left", borderpad=3)

# Plot on the inset
for time_data, auc_data in graph_data:
    # Focus on time range 24 to 72 hours
    mask = (time_data >= 24) & (time_data <= 72)
    ax_inset.plot(time_data[mask], auc_data[mask])

# Customize the inset plot
ax_inset.set_xlim(24, 72)
# ax_inset.set_ylim(0, max([max(auc) for time, auc in graph_data if (time >= 24).all() and (time <= 72).all()]) + 10)
ax_inset.set_title("Focus: 24-72 Hours", fontsize=10)
# ax_inset.set_xlabel("Time (hours)", fontsize=8)
# ax_inset.set_ylabel("Cumulative AUC", fontsize=8)
ax_inset.tick_params(axis='both', which='major', labelsize=8)

plt.savefig("cumulative_icp_Inset.png", dpi=600)
plt.show()

# %%

# NOTE: median calculation lines based on good and bad outcomes 

# comp outcome 0 is alive, 1 is expired 
    # segment dataset based off of these numbers 

# same algo on both segments which will be graphed on the same interface
# for each hour, calculate the corresponding median for AUCHr 
    # range of error for median calculation. point above point below 
# save into list to plot; will contain information on median point and 
# 'error bars' 

# Segment the dataset based on `comp_outcome`
alive_data = df[df['CompOutcome'] == 0]
expired_data = df[df['CompOutcome'] == 1]

# Function to calculate median and error bars for AUCHr per hour
def calculate_median_and_error(data, time_column='Hour', value_column='AUCHr'):
    # Group by time and calculate the median, 25th, and 75th percentiles
    grouped = data.groupby(time_column)[value_column].agg(['median', 'quantile'])
    grouped['q25'] = data.groupby(time_column)[value_column].quantile(0.25)
    grouped['q75'] = data.groupby(time_column)[value_column].quantile(0.75)
    
    # Extract time, median, and error ranges
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
plt.title("Median AUCHr per Hour with Error Bars")
plt.xlabel("Time (hours)")
plt.ylabel("AUCHr")
plt.legend()
plt.grid(True)
plt.xlim(0, 168)  # Adjust as necessary based on your dataset

# Save and display the plot
plt.savefig("median_auchr_plot.png", dpi=600)
plt.show()

# %%
# NOTE: Now every 24 hours. 
# Segment the dataset based on `comp_outcome`
alive_data = df[df['CompOutcome'] == 0]
expired_data = df[df['CompOutcome'] == 1]

# Function to calculate median and error bars for AUCHr every 24 hours
def calculate_median_and_error(data, time_column='Hour', value_column='AUCHr', interval=24):
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
plt.title("Median AUCHr per 24-Hour Interval with Error Bars")
plt.xlabel("Time (hours)")
plt.ylabel("AUCHr")
plt.legend()
plt.grid(True)
plt.xticks(time_alive, rotation=45)  # Ensure x-axis ticks match 24-hour bins
plt.xlim(0, 192)  # Adjust as necessary based on your dataset

# Save and display the plot
plt.savefig("median_auchr_24hr_plot.png", dpi=600)
plt.show()
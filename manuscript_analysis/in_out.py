# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# %%

df_inout = pd.read_csv(r"/home/rayan/Research/medical_proj/ICP_modeling_paper/Older files/InOutP.csv")
df_vitals = pd.read_csv(r"/home/rayan/Research/medical_proj/ICP_modeling_paper/Older files/older_graphing/vitalsP_cleaned.csv")
# %%

# NOTE: lets do some plots first 

df_inout = df_inout[df_inout['intakeoutputoffset'] <= 10080]

# %%

print(f"Before conversion: Min = {df_inout['intakeoutputoffset'].min()} | Max = {df_inout['intakeoutputoffset'].max()}")
df_inout['intakeoutputoffset'] = df_inout['intakeoutputoffset'] / 60
print(f"After conversion: Min = {df_inout['intakeoutputoffset'].min()} | Max = {df_inout['intakeoutputoffset'].max()}")

# %%

df_inout.head()

# %%

merged_df = pd.merge(df_inout, df_vitals, left_on='intakeoutputoffset', right_on='Time')
merged_df.head()

# %%
# NOTE: making sure they are the same

set1 = set(merged_df['intakeoutputoffset'])
set2 = set(merged_df['Time'])

print(set1 - set2)

# %%
merged_df = merged_df[['intakeoutputoffset', 'icp', 'nettotal']]
merged_df['intakeoutputoffset'] = merged_df['intakeoutputoffset'].astype(int) 

aggregated_df = merged_df.groupby('intakeoutputoffset', as_index=False).mean()

plt.figure(figsize=(10,6))
sns.regplot(x='nettotal', y='icp', data=aggregated_df, scatter_kws={'s': 10}, line_kws={'color': 'red'})
# sns.scatterplot(x='nettotal', y='icp', data=aggregated_df)

plt.title("ICP vs nettotal")
# plt.ylim(top=20)
plt.show()


# %%

corr_matrix = merged_df.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# %%

aggregated_df['cumulative_nettotal'] = aggregated_df['nettotal'].cumsum()

plt.figure(figsize=(10, 6))
sns.lineplot(x='intakeoutputoffset', y='cumulative_nettotal', data=aggregated_df)

plt.title("Cumulative Nettotal vs Time")
plt.xlabel("Time (Hours)")
plt.ylabel("Cumulative Nettotal")
plt.show()


# %%

# NOTE: cumulative plot stuff, make sure if this is right. 
df_inout = df_inout.sort_values(by=['patientunitstayid', 'intakeoutputoffset'])
df_inout['cumulative_nettotal'] = df_inout.groupby('patientunitstayid')['nettotal'].cumsum()

# Lists of expired and alive patients
expiredID_list = [306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597]

aliveID_list = [193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239]


expired_data = df_inout[df_inout['patientunitstayid'].isin(expiredID_list)]
alive_data = df_inout[df_inout['patientunitstayid'].isin(aliveID_list)]

# Function to calculate median and error bars for AUCHr every 24 hours
def calculate_median_and_error(data, time_column='intakeoutputoffset', value_column='cumulative_nettotal', interval=24):
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
plt.title("Cumulative nettotal vs Time per 24 hour interval")
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative fluid")
plt.legend()
plt.grid(True)
plt.xticks(time_alive, rotation=45)  # Ensure x-axis ticks match 24-hour bins
# plt.xlim(0, 192)  # Adjust as necessary based on your dataset

# Save and display the plot
# plt.savefig("median_Sp20AUCCum_24hr_plot.png", dpi=600)
# plt.show()
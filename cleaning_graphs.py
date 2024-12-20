# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression
import missingno as mno
from sklearn.preprocessing import MinMaxScaler
#from statsmodels.imputation.mice import MICEData

import plotly.graph_objects as go
import streamlit as st
from LinkedListClass import Node, LL

# %%
df_apache = pd.read_csv('/app/csv_results/apache_results.csv')
df_diagP = pd.read_csv('/app/csv_results/diagP_results.csv')
df_infs = pd.read_csv('/app/csv_results/infsP_results.csv')
df_labsP = pd.read_csv('/app/csv_results/labsP_results.csv')
df_examP = pd.read_csv('/app/csv_results/examP_results.csv')
df_vitalsP = pd.read_csv('/app/csv_results/vitalsP.csv')

icp_list = df_vitalsP['icp']
time_list = df_vitalsP['Time']
df_vitalsP.head()

df_vitalsP = df_vitalsP.drop(columns=['Unnamed: 0', 'observationoffset', 'Day', 'Hour', 'systemicdiastolic', 'systemicsystolic'])
# patient 1082792 literally has only one data point :) 
# missing ids from og list but not in gen list {2782239}
patient_list = [
    306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597, 193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239 
]

df_vitalsP = df_vitalsP.loc[df_vitalsP['patientunitstayid'].isin(patient_list)]
df_vitalsP.head()
# %%

df_vitalsP = df_vitalsP.sort_values(by=['patientunitstayid', 'Time'])
df_vitalCopy = df_vitalsP
df_vitalsP = df_vitalsP.set_index(['patientunitstayid', 'Time'])

# %%
unique_patient_ids = df_vitalsP.index.get_level_values('patientunitstayid').unique()
print(unique_patient_ids)
print(len(unique_patient_ids))

orig_set = set(patient_list)
gen_set = set(unique_patient_ids)
missing_in_generated = orig_set - gen_set
print(f"missing ids from og list but not in gen list {missing_in_generated}")

# %% 

dfL_vitals = LL()

for patient_id in unique_patient_ids: 
    # dfIter = df_vitalsP.loc[patient_id]
    # should get datframe for each patient
    dfIter = df_vitalsP.xs(patient_id, level='patientunitstayid', drop_level=False)
    # dfIter.index.set_names(['patientunitstayid', 'Time'], inplace=True)
    dfL_vitals.append(dfIter)

dfL_vitals.display()
print(dfL_vitals.length())

# %% 

expired_list = [306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597]

alive_list = [193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239]

df_expired = df_vitalCopy[df_vitalCopy['patientunitstayid'].isin(expired_list)]
df_alive = df_vitalCopy[df_vitalCopy['patientunitstayid'].isin(alive_list)]

# %%
# NOTE: Implementing streamlit graphs
# TODO: Switch from streamlit to "snapshot" visualizations (no need for interaction)

# %%
st.title('All ICP values vs Time')
fig = go.Figure()
for patient_id in df_vitalCopy['patientunitstayid'].unique():
    patient_data = df_vitalCopy[df_vitalCopy['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines+markers', name=f'Patient {patient_id}'))

fig.update_layout(title='All ICP values vs Time', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[0, 50])
st.plotly_chart(fig)

# %%
# Bell curve histogram 
with st.expander(f'Histogram of ICP Values for All Patients'):
    # Combine ICP values from all patients
    all_icp_values = df_vitalCopy['icp']

    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=all_icp_values, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Histogram of ICP Values for All Patients',
                    xaxis_title='ICP Value',
                    yaxis_title='Count',
                    bargap=0.2)

    st.plotly_chart(fig)

# %%

st.title('Interactive ICP of Alive patients')


fig = go.Figure()

for patient_id in df_alive['patientunitstayid'].unique():
    patient_data = df_alive[df_alive['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))

fig.update_layout(title='ICP Values of Alive Patients', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[5, 55])

st.plotly_chart(fig)


with st.expander(f'Histogram of ICP Values for Alive Patients'):
    # Combine ICP values from all patients
    all_icp_values = df_alive['icp']

    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=all_icp_values, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Histogram of ICP Values for Alive Patients',
                    xaxis_title='ICP Value',
                    yaxis_title='Count',
                    bargap=0.2)

    st.plotly_chart(fig)
    
    multi = ''' We noticed a :blue-background[normal, right skewed distribution curve] as expected for alive patients.  
    '''

    st.markdown(multi)
# -------

st.title('Interactive ICP of Expired Patients')

fig = go.Figure()

for patient_id in df_expired['patientunitstayid'].unique():
    patient_data = df_expired[df_expired['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))

fig.update_layout(title='Interactive ICP of Expired Patients', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[5, 55])

st.plotly_chart(fig)


with st.expander(f'Histogram of ICP Values for Expired Patients'):
    # Combine ICP values from all patients
    all_icp_values = df_expired['icp']

    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=all_icp_values, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Histogram of ICP Values for Expired Patients',
                    xaxis_title='ICP Value',
                    yaxis_title='Count',
                    bargap=0.2)

    st.plotly_chart(fig)

    multi = ''' However, for expired patients we noticed :blue-background[survivorship bias]. We would have to split by time.  
    We also do note that the mean for this data is to the right of the previous graph for higher ICP values on average. 
    '''

    st.markdown(multi)

# %%

st.title('Vitals of Every Patient')

tempNode = dfL_vitals.head
while tempNode: 
    dt = tempNode.data
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]
    time = dt.index.get_level_values('Time')
    with st.expander(f'Patient ID: {patient}'):
        fig = go.Figure()
        numeric_columns = dt.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column != 'Time':  # Exclude 'Time' as it's our x-axis
                fig.add_trace(go.Scatter(x=time, y=dt[column], mode='lines', name=column))
        
        fig.update_layout(title = f'Patient ID {patient}', xaxis_title = 'Time', yaxis_title = 'Value', hovermode = 'closest')

        # see if we need this 
        # fig.update_yaxes(range=[dt[numeric_columns].min().min(), dt[numeric_columns].max().max()])
        
        if(dt.index.get_level_values('Time').max() > 72):
            fig.update_xaxes(range=[0,72])

        st.plotly_chart(fig)
    
    tempNode = tempNode.next

#%%
df_vitalsP.head()

# %%
def process_icp_range(df_vitalsP, time_col, icp_col, min_icp, max_icp):
    # Create ICP range column
    range_col = f'icpSpike{min_icp}to{max_icp}'
    df_vitalsP[range_col] = df_vitalsP[icp_col].where((df_vitalsP[icp_col] >= min_icp) & (df_vitalsP[icp_col] <= max_icp))
    # Prepare data
    df_clean = df_vitalsP.reset_index()
    df_clean = df_clean[[time_col, range_col]].dropna().reset_index(drop=True)
    time_clean = df_clean[time_col].values
    icp_clean = df_clean[range_col].values
    # Calculate area
    area = np.trapz(icp_clean, time_clean)
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_clean, icp_clean)
    plt.xlabel('Time')
    plt.ylabel(f'ICP ({min_icp}-{max_icp} range)')
    plt.title(f'ICP values in {min_icp}-{max_icp} range')
    # Add area and data points info to plot
    total_points = len(df_vitalsP)
    clean_points = len(time_clean)
    percentage_used = (clean_points / total_points) * 100
    plt.text(0.05, 0.95, f'Area: {area:.2f}\nData points: {clean_points}/{total_points} ({percentage_used:.2f}%)',
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.show()
    return area, clean_points, total_points
# Define ICP ranges
icp_ranges = [(0, 20), (20, 25), (25, 30), (30, 40), (40, 50)]
# Process each range
results = []
for min_icp, max_icp in icp_ranges:
    area, clean_points, total_points = process_icp_range(df_vitalsP, 'Time', 'icp', min_icp, max_icp)
    results.append({
        'range': f'{min_icp}-{max_icp}',
        'area': area,
        'clean_points': clean_points,
        'total_points': total_points,
        'percentage_used': (clean_points / total_points) * 100
    })
# Print summary
for result in results:
    print(f"ICP Range {result['range']}:")
    print(f"  Estimated area: {result['area']:.2f}")
    print(f"  Data points: {result['clean_points']}/{result['total_points']} ({result['percentage_used']:.2f}%)")
    print()

# %%
maximum_icp = df_vitalCopy['icp'].max()

# before we added (35,maximum_icp)
icp_ranges = [(0, 20), (20, 25), (25, 30), (30, 35), (35, maximum_icp)]
# Using linked list code to make our lives easier 

def func_icp_range(DF):
    df_icp_ranges = LL()
    for df_value in icp_ranges: 

        min_icp = df_value[0]
        max_icp = df_value[1]  
        
        # Filter corresponding time column based on the same condition
        filtered_df = DF.loc[(DF['icp'] >= min_icp) & (DF['icp'] <= max_icp), ['icp', 'Time']]

        # Create a new DataFrame with filtered data
        df_icp_ranges.append(filtered_df)
    return df_icp_ranges

df_vitalCopySORTED = df_vitalCopy.sort_values(by=['Time'])

df_icp_ranges = func_icp_range(df_vitalCopy)
df_icp_ranges.display()
print(df_icp_ranges.length())

# %%

def range_traversal(df_icp_ranges):
    tempNode = df_icp_ranges.head
    count = 0
    sumTotal = 0
    while tempNode: 
        dt = tempNode.data
        
        freq = dt['icp'].sum()
        
        range_check = icp_ranges[count]
        
        # y should be first for calculating area under the 
        # curve. trapezoidal riemann
        ipc_load = np.trapz(dt['icp'], dt['Time'])
        sumTotal += ipc_load
        
        print(f"For range {range_check }, frequency is {freq} and ipc_load is {ipc_load}")

        count += 1
        tempNode = tempNode.next
    print(f"THE ACTUAL TOTALED SUM IS {sumTotal}")

range_traversal(df_icp_ranges)

# %%

# use both functions for each patient! 

tempNode = dfL_vitals.head

while tempNode: 
    dt = tempNode.data
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]
    # time = dt.index.get_level_values('Time')

    dt = dt.reset_index()
    # total_ipc_area = np.trapz(dt['icp'], dt['Time'])    

    # print(f'The total ipc area for this patient is {total_ipc_area}')
    dt_linked_list = func_icp_range(dt)

    # This is our print function 
    print(f"For patient {patient}")
    range_traversal(dt_linked_list)
    print("\n")

    tempNode = tempNode.next

# %%

time_ranges = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144), (144, 168)]

def day_icp_load(patient_df, patient):
    df_time_ranges = LL()
    df_icp_loads = LL()
    for df_value in time_ranges: 

        min_time = df_value[0]
        max_time = df_value[1]  
        
        # Filter corresponding time column based on the same condition
        df_day = patient_df.loc[(patient_df['Time'] >= min_time) & (patient_df['Time'] <= max_time), ['icp', 'Time']]     

        df_time_ranges.append(df_day)
    
    # now use df_time_ranges

    icp_load = 0

    tempNode = df_time_ranges.head
    while tempNode:
        dt = tempNode.data
        dt = dt.sort_values(by='Time')
        icp_load = np.trapz(dt['icp'], dt['Time'])
        # append to the actual linked list 
        df_icp_loads.append(icp_load)
        tempNode = tempNode.next

    return df_icp_loads


# now traverse through linked list of patients and calcualte linked list of icp loads for each

patient_list = []
Time_list = []
# currently 7 days 
day_list = [[], [], [], [], [], [], [], []]


tempNode = dfL_vitals.head

while tempNode: 
    dt = tempNode.data
    dt = dt.reset_index()
    patient_list.append(dt['patientunitstayid'].iloc[0])
    time = dt['Time']

    # icp load list, then iterate through the linked list, adding each as its own column
    icp_load_list = LL()
    icp_load_list = day_icp_load(dt, patient)

    tempNode_icp = icp_load_list.head
    count = 0

    while tempNode_icp:
        day_list[count].append(tempNode_icp.data) 
        tempNode_icp = tempNode_icp.next
        count += 1

    sum_area = np.trapz(dt['icp'], dt['Time'])
    day_list[7].append(sum_area)

    tempNode = tempNode.next

data = {
    'patientunitstayid' : patient_list, 
    'Day 1' : day_list[0],
    'Day 2' : day_list[1],
    'Day 3' : day_list[2],
    'Day 4' : day_list[3],
    'Day 5' : day_list[4],
    'Day 6' : day_list[5],
    'Day 7' : day_list[6], 
    'Summation' : day_list[7]
}

df_range = pd.DataFrame(data)
df_range.head(100000000000000000000000000000000000000)

# %%

# -------------------------------------- Pseudocode --------------------------------------

# all patients (node in vitalsP_LL)
    # each patient (node.data (vitalsP for that ID))
        # each day (create 7 days from node.data)
            # each point (find new points, spike count, spike times, spike durations)

                # save new points, spike count, spike times, spike durations into a dataframe
                # save dataframe into a totalDay_List (day 1-7)
                # reset variables for next day
            # save totalDay_list into a totalPatient_list (in same order as vitalsP_LL)


# if from threshold to range, count as spike

node = vitalsP_imputed_LL.head
numNode = 0
while node:
    print(f'Node {numNode}')
    df = node.data # datafarame per ID 
    ID = df['patientunitstayid'].iloc[0] # patient ID
    # saves OG points (day 1-7)
    day1_Graph = df.loc[df['Time'] < 24]
    day2_Graph = df.loc[(df['Time'] >= 24) & (df['Time'] < 48)]
    day3_Graph = df.loc[(df['Time'] >= 48) & (df['Time'] < 72)]
    day4_Graph = df.loc[(df['Time'] >= 72) & (df['Time'] < 96)]
    day5_Graph = df.loc[(df['Time'] >= 96) & (df['Time'] < 120)]
    day6_Graph = df.loc[(df['Time'] >= 120) & (df['Time'] < 144)]
    day7_Graph = df.loc[(df['Time'] >= 144) & (df['Time'] < 168)]
    # saves into a list
    DPList = [day1_Graph, day2_Graph, day3_Graph, day4_Graph, day5_Graph, day6_Graph, day7_Graph]
    # reset day for next patient
    day = 0

    for graph in DPList: # each graph, by day
        day += 1
        newGraph = pd.DataFrame(columns=['patientunitstayid', 'Day', 'Time', 'icp']) # holds points
        spikeStats = pd.DataFrame(columns=['patientunitstayid', 'Day', 'spikes', 'spikeStarts', 'spikeEnds', 'spikeDurations']) # holds lists
        
        spike_Count20, spike_Count25, spike_Count30, spike_Count35 = 0, 0, 0, 0
        start20, start25, start30, start35 = [], [], [], []
        end20, end25, end30, end35 = [], [], [], []
        
        newGraph_days = []
        spikeStats_days = []
        for i in range(len(graph)-1): # each point in graph
            
            # sets current and next point (used for conditions)
            nT = graph['Time'].iloc[i]
            nI = graph['icp'].iloc[i]
            nxT = graph['Time'].iloc[i+1]
            nxI = graph['icp'].iloc[i+1]
            
            # append the current point to graph
            newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': nT, 'icp': nI}, index=[0])
            # Check if newRow is not empty or all-NA before concatenation
            if not newRow.isna().all(axis=None):
                newGraph = pd.concat([newGraph, newRow], ignore_index=True)


            # sets threshold conditions
            if(nI == nxI): # skips if equal icp's
                continue
            for i in range(len(thresholds)): # when threshold is crossed, set condition
                if((nI < thresholds[i] and thresholds[i] < nxI) or (nI > thresholds[i] and thresholds[i] > nxI)):
                    t_cond[i] = True
            
            # finds slope
            slope = (nxI - nI) / (nxT - nT)

            # plotting new points
            # crosses 20
            if(t_cond[0]):
                x = ((20-nI)/slope) + nT
                
                # add new point to graph
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': x, 'icp': 20}, index=[0])
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)
            # crosses 25
            if(t_cond[1]):
                x = ((25-nI)/slope) + nT
                
                # add new point to graph
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': x, 'icp': 25}, index=[0])
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)
            # crosses 30
            if(t_cond[2]):
                x = ((30-nI)/slope) + nT

                # add new point to graph
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': x, 'icp': 30}, index=[0])
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)
            # crosses 35
            if(t_cond[3]):
                x = ((35-nI)/slope) + nT

                # add new point to graph
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': x, 'icp': 35}, index=[0])
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)
            
            ####### SPIKE COUNT
                # manually count when the graphs crosses a 
                # threshold AND include values 
                # where the current point is on the 
                # threshold and passes into the threshol

            # reset condiitons to prep for next point
            t_cond[0] = False
            t_cond[1] = False
            t_cond[2] = False
            t_cond[3] = False

        
        spikes = [spike_Count20, spike_Count25, spike_Count30, spike_Count35]
        # start = [start20, start25, start30, start35]
        # end = [end20, end25, end30, end35]
        # spikeStats = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'spikes': spikes, 'spikeStarts': start, 'spikeEnds': end})
        
        # # sum total spikes
        # print('spike count 20:', spike_Count20)
        # print('spike count 25:', spike_Count25)
        # print('spike count 30:', spike_Count30)
        # print('spike count 35:', spike_Count35)
        # # Assuming newGraph is your DataFrame containing the data
        # plt.figure(figsize=(10, 6))
        # # Plotting the graph
        # plt.plot(graph['Time'], graph['icp'], marker='o', linestyle='-')
        # # Adding horizontal lines at specific icp values
        # plt.axhline(y=20, color='r', linestyle='--', label='Threshold 20')
        # plt.axhline(y=25, color='g', linestyle='--', label='Threshold 25')
        # plt.axhline(y=30, color='b', linestyle='--', label='Threshold 30')
        # plt.axhline(y=35, color='y', linestyle='--', label='Threshold 35')
        # # Adding title and labels
        # plt.title(f'{ID}, Day {day}: ICP vs Time')
        # plt.xlabel('Time')
        # plt.ylabel('ICP')
        # # Adding legend
        # plt.legend()
        # # Display the plot
        # plt.show()

        # resets conditions for next day
        spike_Count20, spike_Count25, spike_Count30, spike_Count35 = 0, 0, 0, 0
        start20, start25, start30, start35 = [], [], [], []
        end20, end25, end30, end35 = [], [], [], []

        # print(newGraph)
        # graphs and stats for all 7 days stored into a df
        # newGraph_days.append(newGraph)
        # spikeStats_days.append(spikeStats)
    numNode += 1
    node = node.next


# %%

# Thresholds for ICP ranges
thresholds = [20, 25, 30, 35]
less_20_list = []


# Function to shift ICP values
def shift_icp_values(df, shift_amount):
    df['icp'] = df['icp'] - shift_amount
    df = df[df['icp'] >= 0]
    return df

def less20_area(df):
    df = df.loc[df['icp'] <= 20, ['icp', 'Time']]
    return df

# Function to calculate AUC for given thresholds
def calc_auc(df):
    results = []
    for threshold in thresholds: 
        data_above_threshold = df.loc[df['icp'] >= threshold, ['icp', 'Time']]
        data_above_threshold = shift_icp_values(data_above_threshold, threshold)
        if not data_above_threshold.empty:
            x = data_above_threshold['Time'].values
            y = data_above_threshold['icp'].values
            area = np.trapz(y, x)
            results.append(area)
        else:
            results.append(0)
    return results

patient_list = []
auc_list = [[] for _ in range(len(thresholds))]
test_list = []

for df in plotPointsNew_List:
    # Ensure df is not a Series and not empty
    if not isinstance(df, pd.Series) and (len(df) != 0):
        patient_id = df['patientunitstayid'].iloc[0]
        
        # Append to the list         
        patient_list.append(patient_id)

        auc_result = calc_auc(df)

        for i in range(len(thresholds)):
            auc_list[i].append(auc_result[i])
        test_list.append(np.trapz(df['icp'], df['Time']))

        test_area = less20_area(df)
        less_20_list.append(np.trapz(test_area['icp'], test_area['Time']))
        
# Create the DataFrame
data = {
    'patientunitstayid': patient_list, 
    '>20': auc_list[0],
    '>25': auc_list[1],
    '>30': auc_list[2],
    '>35': auc_list[3], 
    'Total (tested)': test_list,
    '<20' : less_20_list
}

df_auc_ranges = pd.DataFrame(data)

print(f"Number of plot points: {len(plotPointsNew_List)}")
df_auc_ranges.head(10000000000000000000000000000000000000000)

# %% 

# ------------------ EXPERIMENTS -----------------
# export vitalsPcopy and this threshold. drop temperature for now. 
 
df_vitalCopy.head()

print(df_vitalCopy.isna().sum())

df_vitalCopy.to_csv('EXPERIMENTS_AAHH.csv')
df_auc_ranges.to_csv('EXPERIMENTS_AUC_RANGES.csv')
df_range.to_csv('EXPERIMENTS_RANGE.csv')
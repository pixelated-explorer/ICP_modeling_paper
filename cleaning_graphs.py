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


# -*- coding: utf-8 -*-
"""
Simulate gang involvement data
"""

import pandas as pd
import numpy as np
import stats.synthesis as synth
import stats.shuffle as shuffle

data = pd.read_excel("A:/UCLMaster_V2.xlsx", sheet_name=1, index_col='URN')

# Count how many valid values for each time period
data['ValidPre'] = data.loc[:,'Pre48':'Pre1'].count(axis=1)
data['ValidDur'] = data.loc[:,'Dur1':'Dur63'].count(axis=1)
data['ValidPost'] = data.loc[:,'Post1':'Post48'].count(axis=1)
data['ValidAll'] = data.loc[:,'Pre48':'Post48'].count(axis=1)

# Fill missing values of PostMths with zeros
data['PostMths'].fillna(0, inplace=True)
data['PreDurMths'] = data['PreMths'] + data['DuringMths']

# Note the first valid month in the data, measured from 10th birthday
# Month columns are measured SINCE this - first valid PreXX is 0 months after StartMth
data['StartMth'] = data['PreMths'] - data['ValidPre']

# EndMth will be one after the last valid month in the data (i.e. the next missing month)
data['EndMth'] = data['StartMth'] + data['ValidAll']

histories_raw = data.loc[:, 'Pre48': 'Post48'].T

histories_continuous = {}
for urn in histories_raw:
    history = histories_raw[urn].dropna()
    history.index = np.arange(data.loc[urn,'StartMth'], data.loc[urn,'EndMth'])
    histories_continuous[urn] = history
histories = pd.DataFrame(histories_continuous)

sanction_rates = histories.mean(axis=1)
entry_points = data['PreMths']


# n_people = 1000
n_perm = 99
n_runs = 100

config_results = []

for n_people in np.linspace(500, 4000, 8, dtype=np.int64):
    for treatment_factor in np.linspace(0.6, 0.9, 7):
        pvals = np.empty((n_runs, 48))
        
        for run in range(n_runs):
            
            print(f'n_people: {n_people} - treatment_factor: {treatment_factor} - run: {run}')
        
            entry_points_synth = synth.sample_entry_points(entry_points, n_people)
            histories_synth = synth.simulate_histories(sanction_rates, n_people, treatment_factor, entry_points_synth)
            
            histories_aligned = shuffle.align_data_buff(histories_synth, entry_points_synth, 24, 24)
            observed_rates = histories_aligned.mean(axis=1)
            
            perm_rates = shuffle.permuted_rates(histories_synth, entry_points_synth, 24, 24, n_perm)
            
            stacked = np.vstack((observed_rates, perm_rates))
            indices = np.argsort(stacked, axis=0)
            ranks = np.argmax(indices == 0, axis=0)
            
            pvals[run, :] = (ranks + 1) / (n_perm + 1)
        
        pvals = pd.DataFrame(pvals, columns=np.arange(-24, 24))
        pvals['n_people'] = n_people
        pvals['treatment_factor'] = treatment_factor
        pvals['n_perm'] = n_perm
        pvals['run'] = np.arange(n_runs)
        
        config_results.append(pvals)

results = pd.concat(config_results, ignore_index=True)

results.to_csv('./output/power_outputs.csv')



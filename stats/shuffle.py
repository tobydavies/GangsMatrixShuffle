# -*- coding: utf-8 -*-
"""
Permutation tests
"""

import pandas as pd
import numpy as np


def align_data(histories, shifts):
    # Approach is based on 'rolling' each column upwards
    # This means that the values that get shifted off the top reappear at the bottom
    # Thanks to https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently/20361561#20361561
    # Get open meshgrid containing row and column indices
    rows, cols = np.ogrid[:histories.shape[0], :histories.shape[1]]
    # Use broadcasting logic to get different shifts of the row indices on each column
    # Basically creates multiple copies of the row indices, subtracting different amount each time
    row_indices = rows - (histories.shape[0] - shifts.values)
    # Get aligned histories by using these indices to slice the original values
    aligned_histories = histories.values[row_indices, cols]
    return aligned_histories


def align_data_buff(histories, shifts, pre_steps, post_steps):
    # Realign the data by 'rolling' columns upwards
    hist = align_data(histories, shifts)
    # The 'pre' period now appears at the bottom of the array, so slice this
    pre_data = hist[-pre_steps:,:]
    # The 'post' steps are at the top of the array, so slice these
    post_data = hist[:post_steps,:]
    # Stack these vertically
    hist = np.vstack((pre_data, post_data))
    # Return the resulting DataFrame with row indices set accordingly
    return pd.DataFrame(hist, columns=histories.columns, index=np.arange(-pre_steps, post_steps))


def permuted_rates(histories, shifts, pre_steps, post_steps, n_perm):
    #Make a copy of the shifts for shuffling
    shifts_perm = shifts.copy()
    #Initialise a list to store results from each shuffle
    perm_rates = []
    #Repeat for a number of permutations
    for i in range(n_perm):
        #Shuffle the shift offsets across individuals
        shifts_perm.loc[:] = shifts_perm.sample(frac=1).values
        #Calculate aligned histories using the shuffled offsets
        histories_aligned = align_data_buff(histories, shifts_perm, pre_steps, post_steps)
        #Calculate sanction rates for each month
        rates = histories_aligned.mean(axis=1)
        #Add to the list
        perm_rates.append(rates)
    return pd.DataFrame(perm_rates)


def permutation_test(histories, shifts, pre_steps, post_steps, n_perm):
    # Compute aligned histories
    histories_aligned = align_data_buff(histories, shifts, pre_steps, post_steps)
    # Calculate observed sanction rates
    observed_rates = histories_aligned.mean(axis=1)
    # Calculate rates under shuffling
    perm_rates = permuted_rates(histories, shifts, pre_steps, post_steps, n_perm)
    # Stack observed rates over permuted rates
    stacked = np.vstack((observed_rates, perm_rates))
    # Get indices required to sort
    indices = np.argsort(stacked, axis=0)
    # P-value corresponds to the position of 0 in the indices list, divided by number of shuffles
    pvals = np.argmax(indices == 0, axis=0) / (n_perm + 1)
    # Structure the p-values as a monthly time series
    pvals = pd.Series(pvals, index=np.arange(-pre_steps, post_steps+1))
    return pvals
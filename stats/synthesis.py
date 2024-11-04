# -*- coding: utf-8 -*-
"""
Simulate gang sanction data
"""

import pandas as pd
import numpy as np


def sample_entry_points(entry_points, n, seed=None):
    # Sample with replacement from a Pandas Series containing entry points as months
    sample = entry_points.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    return sample


def simulate_histories(sanction_rates, n, treatment_factor, entry_points, seed=None):
    # Replicate the observed sanction rates n times, producing a column for each individual
    sanction_rates_all = np.tile(sanction_rates.values[:, np.newaxis], (1, n))
    # Get the indices for the months and individuals
    month_index, person_index = np.indices(sanction_rates_all.shape)
    # Create a mask - for each column, values are True for rows higher than corresponding entry point 
    mask = month_index >= entry_points.values
    # Multiply the post-entry sanction rates by the treatment factor
    sanction_rates_all[mask] *= treatment_factor
    # Initialise RNG
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()
    # Generate synthetic histories using Poisson distribution with specified rates
    histories = rng.poisson(sanction_rates_all) > 0
    # Convert this to DataFrame
    histories = pd.DataFrame(histories.astype(int))
    return histories
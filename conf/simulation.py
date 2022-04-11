import numpy as np
from conf.conf import noise_level

# Definition of hidden nodes
observed_nodes = [1, 3, 4, 6, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 24, 26, 28,
                  32, 36, 37, 39, 40, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55]  # About 60% of the nodes
hidden_nodes = list(set(range(56-1)) - set(np.array(observed_nodes)-1))
#hidden_nodes = []

# Assumption of constant Z loads on hidden nodes
constant_load_hidden_nodes = True

# Week during which the sample size starts
selected_weeks = np.array([0])  # 0 is week 12 for the small dataset. Replace by 12 of using the full year data.

# Total sample size on which block averaging is performed
days = 7*len(selected_weeks)

# Number of averaged blocks
time_steps = 24 * 60 * 7  # 15000 is max capable for GPU RTX 3090

# Sensor sampling frequency
measurement_frequency = 50 # [Hz] # 50

# Safety factor for sensor dimension (compared to nominal node power)
safety_factor = 4

# Do not touch that, rather change noise_level in conf.conf
current_magnitude_sd = 1e-4 * noise_level
voltage_magnitude_sd = 1e-4 * noise_level
current_phase_sd = 0.01*np.pi/180 * noise_level
voltage_phase_sd = 0.01*np.pi/180 * noise_level

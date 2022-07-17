import numpy as np
from conf.conf import noise_level

# Definition of observed nodes for ieee123bus network
list1 = [3, 16, 18, 28, 40, 47, 52]                         #
list2 = [8, 9, 22, 36, 49, 46, 51]                          #
list3 = [6, 12, 55, 26, 44, 50]                             #36% (20 nodes
list4 = [1, 4, 10, 15, 17, 37]                              #47% (26
list5 = [19, 24, 32, 39, 43, 53]                            #57% (32
list6 = [5, 14, 15, 20, 25, 29, 31, 34, 38, 41, 45, 54]     #79%

observed_nodes = list1 + list2 + list3 + list4 + list5 #+ list6
hidden_nodes = list(set(range(1,56)) - set(np.array(observed_nodes)-1))
hidden_nodes = []

# Assumption of constant Z loads on hidden nodes
constant_load_hidden_nodes = True
load_constantness = 1.0 #0.84

# Week during which the sample size starts
selected_weeks = np.array([0])  # 0 is week 12 for the small dataset. Replace by 12 of using the full year data.

# Total sample size on which block averaging is performed
days = 1*len(selected_weeks) #7*len(selected_weeks)

# Number of averaged blocks
time_steps = 24 * 60 #* 7  # 15000 is max capable for GPU RTX 3090

# Sensor sampling frequency
measurement_frequency = 50 # [Hz] # 50

# Safety factor for sensor dimension (compared to nominal node power)
safety_factor = 4

# Do not touch that, rather change noise_level in conf.conf
current_magnitude_sd = 1e-4 * noise_level * 100
voltage_magnitude_sd = 1e-4 * noise_level
current_phase_sd = 0.01*np.pi/180 * noise_level * 100
voltage_phase_sd = 0.01*np.pi/180 * noise_level

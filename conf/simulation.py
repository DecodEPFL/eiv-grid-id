import numpy as np

observed_nodes = [1, 3, 4, 6, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 24, 26, 28,
                  32, 36, 37, 39, 40, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55]  # About 60% of the nodes
observed_nodes = list(range(1, 57))

hidden_nodes = list(set(range(56)) - set(np.array(observed_nodes)-1))
constant_load_hidden_nodes = False


selected_weeks = np.array([12])
days = len(selected_weeks)*30
time_steps = 15000 # 500
load_cv = 0.0
current_magnitude_sd = 1e-4
voltage_magnitude_sd = 1e-4
phase_sd = 1e-4#0.01*np.pi/180
measurement_frequency = 100 # [Hz] # 50
safety_factor = 4

#for 3ph
days = len(selected_weeks)*30
hidden_nodes = []

import numpy as np

from src.simulation.load_profile import BusData
from src.simulation.lines import LineData

"""

    Copyright @donelef, @jbrouill on GitHub
"""


cigre_mv_feeder3_bus = \
    [BusData(1, 1, (0.50170, 0.50170, 0.50170), (0.208882, 0.208882, 0.208882), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(2, 1, (0.43165, 0.43165, 0.43165), (0.108182, 0.108182, 0.108182), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(3, 1, (0.72750, 0.72750, 0.72750), (0.182329, 0.182329, 0.182329), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(4, 1, (0.54805, 0.54805, 0.54805), (0.137354, 0.137354, 0.137354), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(5, 1, (0.07650, 0.07650, 0.07650), (0.047410, 0.047410, 0.047410), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(6, 1, (0.58685, 0.58685, 0.58685), (0.147078, 0.147078, 0.147078), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(7, 1, (0.57375, 0.57375, 0.57375), (0.355578, 0.355578, 0.355578), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(8, 1, (0.54330, 0.54330, 0.54330), (0.161264, 0.161264, 0.161264), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(9, 1, (0.32980, 0.32980, 0.32980), (0.082656, 0.082656, 0.082656), 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8),
     BusData(10, 3, 10.00000, 10.00000, 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8)]

cigre_mv_feeder3_net = \
    [LineData(10, 1, 2*4.42, "UG3"),
     LineData(1, 2, 2*0.61, "UG3"),
     LineData(2, 3, 2*0.56, "UG3"),
     LineData(3, 4, 2*1.54, "UG3"),
     LineData(4, 5, 2*0.24, "UG3"),
     LineData(5, 6, 2*1.67, "UG3"),
     LineData(6, 7, 2*0.32, "UG3"),
     LineData(7, 8, 2*0.77, "UG3"),
     LineData(8, 9, 2*0.33, "UG3"),
     LineData(1, 6, 2*1.30, "UG3"),
     LineData(9, 2, 2*0.49, "UG3")]

test_3phase_bus = \
    [BusData(1, 1, (0.04, 0.02, 0.02), (0.01, 0.02, 0.01), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(2, 1, (0.08, 0.08, 0.08), (0.02, 0.02, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(3, 1, (0.08, 0, 0), (0.04, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(4, 1, (0, 0.12, 0), (0, 0.08, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(5, 3, 10.00000, 10.00000, 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8)]

test_3phase_net = \
    [LineData(5, 1, 0.4, "OG3"),
     LineData(1, 2, 0.2, "OG3"),
     LineData(2, 3, 0.2, "OG3"),
     LineData(2, 4, 0.2, "OG3")]


"""
Similarly to http://home.engineering.iastate.edu/~jdm/ee457/SymmetricalComponents2.pdf
Z_0 = Z_Y/3 + Z_n, Z_1 = Z_Y/3 where diagonal elements are Z_Y + Z_n and off-diagonal at Z_n
Defining Z_Y as Z_S-Z_M and Z_N as Z_M, we get Z_0 = Z_S/3 + 2*Z_M/3 and Z_1 = (Z_S - Z_M)/3

from https://github.com/tshort/OpenDSS/blob/master/Distrib/IEEETestCases/123Bus/IEEELineCodes.DSS
Z_S ≈ 0.29 + 0.14j and Z_M ≈ 0.095 + 0.05j for UG lines
Z_S ≈ 0.087 + 0.20j and Z_M ≈ 0.029 + 0.08j for OG lines

2 phased lines have almost the same parameters as 3 phased and 1 phased is only OG as
Z1_1 = Z1_0 ≈ 0.25 + 0.25j

Therefore:
"""
Z3_0_og = (0.087 + 0.20j)/3 + 2*(0.029 + 0.08j)/3
Z3_1_og = (0.087 + 0.20j)/3 - (0.029 + 0.08j)/3
Z1_1_og = (0.25 + 0.25j)/3
Z1_0_og = Z1_1_og

Z3_0_ug = (0.29 + 0.14j)/3 + 2*(0.095 + 0.05j)/3
Z3_1_ug = (0.29 + 0.14j)/3 - (0.095 + 0.05j)/3

ieee123_types = {"OG3" : {"r_ohm_per_km": np.real(Z3_1_og), "x_ohm_per_km": np.imag(Z3_1_og),
                "c_nf_per_km": 0, "max_i_ka": 10000, "c0_nf_per_km":  0,
                "r0_ohm_per_km": np.real(Z3_0_og), "x0_ohm_per_km": np.imag(Z3_0_og)},

                "UG3" : {"r_ohm_per_km": np.real(Z3_1_ug), "x_ohm_per_km": np.imag(Z3_1_ug),
                "c_nf_per_km": 0, "max_i_ka": 10000, "c0_nf_per_km":  0,
                "r0_ohm_per_km": np.real(Z3_0_ug), "x0_ohm_per_km": np.imag(Z3_0_ug)},

                "OG1" : {"r_ohm_per_km": np.real(Z1_1_og), "x_ohm_per_km": np.imag(Z1_1_og),
                "c_nf_per_km": 0, "max_i_ka": 10000, "c0_nf_per_km":  0,
                "r0_ohm_per_km": np.real(Z1_0_og), "x0_ohm_per_km": np.imag(Z1_0_og)}
}


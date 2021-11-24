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
     BusData(10, 3, 5.00000, 5.00000, 0.000, 0.000, 1, 1, 0, 20, 1, 1.2, 0.8)]

cigre_mv_feeder3_net = \
    [LineData(10, 1, 4.42, "UG3CIGRE"),
     LineData(1, 2, 0.61, "UG3CIGRE"),
     LineData(2, 3, 0.56, "UG3CIGRE"),
     LineData(3, 4, 1.54, "UG3CIGRE"),
     LineData(4, 5, 0.24, "UG3CIGRE"),
     LineData(5, 6, 1.67, "UG3CIGRE"),
     LineData(6, 7, 0.32, "UG3CIGRE"),
     LineData(7, 8, 0.77, "UG3CIGRE"),
     LineData(8, 9, 0.33, "UG3CIGRE"),
     LineData(1, 6, 1.30, "UG3CIGRE"),
     LineData(9, 2, 0.49, "UG3CIGRE")]

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


# Full 56 bus network
ieee123center_bus = \
    [BusData(1, 1, (0.04, 0.02, 0.1), (0.02, 0.01, 0.05), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(2, 1, (0.02, 0, 0), (0.01, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(3, 1, (0.1, 0.02, 0), (0.05, 0.01, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(4, 1, (0, 0, 0.1), (0, 0, 0.05), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(5, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(6, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(7, 1, (0, 0, 0), (0, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(8, 1, (0.02, 0, 0), (0.01, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(9, 1, (0, 0.02, 0), (0, 0.01, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(10, 1, (0, 0.04, 0), (0, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(11, 1, (0.02, 0, 0), (0.01, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(12, 1, (0, 0, 0.04), (0, 0, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(13, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(14, 1, (0, 0.075, 0), (0, 0.035, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(15, 1, (0.035, 0.035, 0.07), (0.025, 0.025, 0.05), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(16, 1, (0, 0, 0.075), (0, 0, 0.035), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(17, 1, (0.12, 0, 0), (0.06, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(18, 1, (0, 0, 0.12), (0, 0, 0.06), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(19, 1, (0.105, 0.07, 0.07), (0.08, 0.05, 0.05), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(20, 1, (0, 0.04, 0), (0, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(21, 1, (0, 0, 0), (0, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(22, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(23, 1, (0, 0.04, 0), (0, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(24, 1, (0, 0, 0.06), (0, 0, 0.03), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(25, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(26, 1, (0, 0, 0.02), (0, 0, 0.01), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(27, 1, (0, 0.02, 0), (0, 0.01, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(28, 1, (0.04, 0.04, 0), (0.02, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(29, 1, (0, 0.04, 0), (0, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(30, 1, (0, 0, 0.04), (0, 0, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(31, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(32, 1, (0, 0.04, 0), (0, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(33, 1, (0, 0, 0), (0, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(34, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(35, 1, (0, 0.04, 0), (0, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(36, 1, (0, 0, 0.04), (0, 0, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(37, 1, (0, 0, 0.1), (0, 0, 0.05), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(38, 1, (0, 0.08, 0), (0, 0.04, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(39, 1, (0.14, 0, 0), (0.07, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(40, 1, (0.08, 0, 0), (0.04, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(41, 1, (0, 0.04, 0), (0, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(42, 1, (0, 0, 0.04), (0, 0, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(43, 1, (0.04, 0, 0.04), (0.02, 0, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(44, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(45, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(46, 1, (0, 0, 0.04), (0, 0, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(47, 1, (0.08, 0.04, 0), (0.04, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(48, 1, (0, 0, 0.02), (0, 0, 0.01), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(49, 1, (0.02, 0.04, 0), (0.01, 0.02, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(50, 1, (0.04, 0, 0), (0.02, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     # technically 3-phase load 0.105, 0.075
     BusData(51, 1, (0.035, 0.035, 0.035), (0.025, 0.025, 0.025), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     # technically 3-phase load 0.210, 0.150
     BusData(52, 1, (0.07, 0.07, 0.07), (0.05, 0.05, 0.05), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(53, 1, (0.035, 0.07, 0.035), (0.025, 0.05, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(54, 1, (0, 0, 0.04), (0, 0, 0.02), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(55, 1, (0.02, 0, 0), (0.01, 0, 0), 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
     BusData(56, 3, 5.000, 5.000, 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8)]

ieee123center_net = \
    [LineData(56, 1, 400*0.3/1000, "OG3"),
     LineData(1, 2, 300*0.3/1000, "OG3"),
     LineData(2, 3, 200*0.3/1000, "OG3"),
     LineData(3, 4, 300*0.3/1000, "OG3"),
     LineData(4, 5, 400*0.3/1000, "OG3"),
     LineData(5, 6, 200*0.3/1000, "OG3"),
     LineData(6, 7, 125*0.3/1000, "OG3"),
     LineData(7, 8, 275*0.3/1000, "OG3"),
     LineData(8, 9, 275*0.3/1000, "OG3"),
     LineData(7, 10, 350*0.3/1000, "OG3"),
     LineData(10, 11, 750*0.3/1000, "OG3"),
     LineData(11, 12, 250*0.3/1000, "UG3"),
     LineData(12, 13, 175*0.3/1000, "UG3"),
     LineData(13, 14, 350*0.3/1000, "UG3"),
     LineData(14, 15, 425*0.3/1000, "UG3"),
     LineData(15, 16, 325*0.3/1000, "UG3"),
     LineData(11, 17, 350*0.3/1000, "OG3"),
     LineData(17, 18, 275*0.3/1000, "OG3"),
     LineData(18, 19, 200*0.3/1000, "OG3"),
     LineData(19, 20, 400*0.3/1000, "OG3"),
     LineData(20, 21, 100*0.3/1000, "OG3"),
     LineData(21, 22, 225*0.3/1000, "OG3"),
     LineData(21, 23, 475*0.3/1000, "OG3"),
     LineData(23, 24, 475*0.3/1000, "OG3"),
     LineData(24, 25, 250*0.3/1000, "OG3"),
     LineData(25, 26, 250*0.3/1000, "OG3"),
     LineData(19, 27, 700*0.3/1000, "OG3"),
     LineData(27, 28, 450*0.3/1000, "OG3"),
     LineData(28, 29, 275*0.3/1000, "OG3"),
     LineData(29, 30, 225*0.3/1000, "OG3"),
     LineData(30, 31, 225*0.3/1000, "OG3"),
     LineData(31, 32, 300*0.3/1000, "OG3"),
     LineData(17, 33, 250*0.3/1000, "OG3"),
     LineData(33, 34, 275*0.3/1000, "OG3"),
     LineData(34, 35, 550*0.3/1000, "OG3"),
     LineData(35, 36, 300*0.3/1000, "OG3"),
     LineData(33, 37, 250*0.3/1000, "OG3"),
     LineData(37, 38, 275*0.3/1000, "OG3"),
     LineData(38, 39, 325*0.3/1000, "OG3"),
     LineData(4, 40, 825*0.3/1000, "OG3"),
     LineData(40, 41, 300*0.3/1000, "OG3"),
     LineData(41, 42, 250*0.3/1000, "OG3"),
     LineData(42, 43, 275*0.3/1000, "OG3"),
     LineData(43, 44, 200*0.3/1000, "OG3"),
     LineData(44, 45, 300*0.3/1000, "OG3"),
     LineData(45, 46, 350*0.3/1000, "OG3"),
     LineData(40, 47, 375*0.3/1000, "OG3"),
     LineData(47, 48, 250*0.3/1000, "OG3"),
     LineData(48, 49, 250*0.3/1000, "OG3"),
     LineData(49, 50, 200*0.3/1000, "OG3"),
     LineData(50, 51, 250*0.3/1000, "OG3"),
     LineData(51, 52, 150*0.3/1000, "OG3"),
     LineData(51, 53, 250*0.3/1000, "OG3"),
     LineData(53, 54, 250*0.3/1000, "OG3"),
     LineData(54, 55, 250*0.3/1000, "OG3")]

"""
Similarly to http://home.engineering.iastate.edu/~jdm/ee457/SymmetricalComponents2.pdf
Z_0 = Z_Y + 3*Z_n, Z_1 = Z_Y where diagonal elements are Z_Y + Z_n and off-diagonal at Z_n
Defining Z_Y as Z_S-Z_M and Z_N as Z_M, we get Z_0 = Z_S + 2*Z_M and Z_1 = (Z_S - Z_M)

from https://github.com/tshort/OpenDSS/blob/master/Distrib/IEEETestCases/123Bus/IEEELineCodes.DSS
Z_S ≈ 0.29 + 0.14j and Z_M ≈ 0.095 + 0.05j for UG lines
Z_S ≈ 0.087 + 0.20j and Z_M ≈ 0.029 + 0.08j for OG lines
[in ohm/kft] (we want km instead so divide by 0.3)

2 phased lines have almost the same parameters as 3 phased and 1 phased is only OG as
Z1_1 = Z1_0 ≈ 0.25 + 0.25j

Therefore:
"""
Z3_0_og = ((0.087 + 0.20j) + 2*(0.029 + 0.08j)) / 0.3
Z3_1_og = ((0.087 + 0.20j) - (0.029 + 0.08j)) / 0.3
Z1_1_og = (0.25 + 0.25j) / 0.3
Z1_0_og = Z1_1_og / 0.3

Z3_0_ug = ((0.29 + 0.14j) + 2*(0.095 + 0.05j)) / 0.3
Z3_1_ug = ((0.29 + 0.14j) - (0.095 + 0.05j)) / 0.3

ieee123_types = {"OG3" : {"r_ohm_per_km": np.real(Z3_1_og), "x_ohm_per_km": np.imag(Z3_1_og),
                "c_nf_per_km": 0, "max_i_ka": 10000, "c0_nf_per_km":  0,
                "r0_ohm_per_km": np.real(Z3_0_og), "x0_ohm_per_km": np.imag(Z3_0_og)},

                "UG3" : {"r_ohm_per_km": np.real(Z3_1_ug), "x_ohm_per_km": np.imag(Z3_1_ug),
                "c_nf_per_km": 0, "max_i_ka": 10000, "c0_nf_per_km":  0,
                "r0_ohm_per_km": np.real(Z3_0_ug), "x0_ohm_per_km": np.imag(Z3_0_ug)},

                "OG1" : {"r_ohm_per_km": np.real(Z1_1_og), "x_ohm_per_km": np.imag(Z1_1_og),
                "c_nf_per_km": 0, "max_i_ka": 10000, "c0_nf_per_km":  0,
                "r0_ohm_per_km": np.real(Z1_0_og), "x0_ohm_per_km": np.imag(Z1_0_og)},

                "UG3CIGRE" : {"r_ohm_per_km": np.real(Z3_1_ug)/3, "x_ohm_per_km": 3*np.imag(Z3_1_ug)/3,
                "c_nf_per_km": 0, "max_i_ka": 10000, "c0_nf_per_km":  0,
                "r0_ohm_per_km": 3*np.real(Z3_0_ug)/3, "x0_ohm_per_km": 3*np.imag(Z3_0_ug)/3}
}


import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from pandapower.timeseries import DFData

"""
    Data class for information on network lines and their admittance.

    Functions to convert between sequence and phase.

    Copyright @donelef, @jbrouill on GitHub
"""

class LineData(object):
    def __init__(self, f: int, t: int, l: float = 1, typ = None, r: float = 0, x: float = 0, b: float = 0,
                 ra: float = 100, rb: float = 100, rc: float = 100, rat: float = 0, a: float = 0, s: int = 1,
                 amin: float = -360, amax: float = 360):
        """
        :param f: index of starting bus
        :param t: index of ending bus
        :param l: length in km
        :param type: name of the standard type, leave None if defining from parameters
        :param r: total resistance in Ohms
        :param x: total reactance in Ohms
        :param b: shunt susceptance in Farads
        :param ra: rate of phase A in percents
        :param rb: rate of phase B in percents
        :param rc: rate of phase B in percents
        :param rat: unused
        :param a: unused
        :param s: status (1 is up, 0 is broken)
        :param amin: minimum phase angle (for securities)
        :param amax: maximum phase angle (for securities)
        """
        self.start_bus = f
        self.end_bus = t
        self.R = r
        self.X = x
        self.B = b
        self.rateA = ra
        self.rateB = rb
        self.rateC = rc
        self.ratio = rat
        self.angle = a
        self.status = s
        self.angmin = amin
        self.angmax = amax
        self.length = l
        self.t = typ

def admittance_sequence_to_phase(y_012):
    """
    Converts a squence admittance matrix (diagonal 3x3 blocks) into a phase one

    :param y_012: Sequence admittance matrix as numpy array
    :return: Admittance matrix as numpy array
    """
    t_abc = np.kron(np.eye(int(y_012.shape[1]/3)), np.array(
        [
            [1, 1, 1],
            [1, np.exp(1j * np.deg2rad(-120)), np.exp(1j * np.deg2rad(120))],
            [1, np.exp(1j * np.deg2rad(120)), np.exp(1j * np.deg2rad(-120))]
        ]))

    t_012 = np.kron(np.eye(int(y_012.shape[1]/3)), np.divide(np.array(
        [
            [1, 1, 1],
            [1, np.exp(1j * np.deg2rad(120)), np.exp(1j * np.deg2rad(-120))],
            [1, np.exp(1j * np.deg2rad(-120)), np.exp(1j * np.deg2rad(120))]
        ]), 3))

    return t_abc @ y_012 @ t_012


def measurement_sequence_to_phase(measurement):
    """
    Converts a squence measurement matrix (diagonal 3x3 blocks) into a phase one

    :param y_012: Sequence measurement matrix as numpy array
    :return: measurement matrix as numpy array
    """
    t_abc = np.kron(np.eye(int(measurement.shape[1]/3)), np.array(
        [
            [1, 1, 1],
            [1, np.exp(1j * np.deg2rad(-120)), np.exp(1j * np.deg2rad(120))],
            [1, np.exp(1j * np.deg2rad(120)), np.exp(1j * np.deg2rad(-120))]
        ]))

    return measurement @ t_abc


def measurement_phase_to_sequence(measurement):
    """
    Converts a phase measurement matrix (diagonal 3x3 blocks) into a sequence one

    :param y_012: measurement matrix as numpy array
    :return: Sequence measurement matrix as numpy array
    """
    t_012 = np.kron(np.eye(int(measurement.shape[1]/3)), np.divide(np.array(
        [
            [1, 1, 1],
            [1, np.exp(1j * np.deg2rad(120)), np.exp(1j * np.deg2rad(-120))],
            [1, np.exp(1j * np.deg2rad(-120)), np.exp(1j * np.deg2rad(120))]
        ]), 3))

    return measurement @ t_012



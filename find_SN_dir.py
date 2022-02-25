# %%
from abc import ABCMeta, abstractmethod

from scipy.optimize import brute, optimize
from scipy.interpolate import interp2d
from scipy.stats import mode
import ROOT
import math
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

# %% PDF definitions


class PDF(metaclass=ABCMeta):
    name: str

    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def eval(self, energy, cos_angle):
        pass


class NumericPDF(PDF):
    """Pointing information about a particular type of interaction (such as electron scattering)."""
    name: str
    energy_min: float
    energy_max: float
    energy_bin_width: float

    cosAngle_min: float
    cosAngle_max: float
    cosAngle_bin_width: float

    interpolation: str | None
    pdf: np.ndarray
    pdf_linear: Callable
    pdf_cubic: Callable

    def __init__(self, name, energy_binning, cosAngle_binning, pdf, interpolation=None):
        super().__init__(name)
        (self.energy_min, self.energy_max, self.energy_bin_width) = energy_binning
        (self.cosAngle_min, self.cosAngle_max,
         self.cosAngle_bin_width) = cosAngle_binning
        self.pdf = pdf
        assert interpolation in (None, 'linear', 'cubic'), print(
            "Interpolation modes are None, linear, cubic")
        self.interpolation = interpolation
        energy_coords = np.arange(0, len(pdf))
        cos_angle_coords = np.arange(0, len(pdf[0]))
        self.pdf_linear = interp2d(energy_coords, cos_angle_coords, np.transpose(pdf),
                                   bounds_error=False, fill_value=None, kind='linear')
        self.pdf_cubic = interp2d(energy_coords, cos_angle_coords, np.transpose(pdf),
                                  bounds_error=False, fill_value=None, kind='cubic')

    def eval(self, energy, cos_angle):
        if energy > self.energy_max:
            return 1.0
        energy_bin = (energy - self.energy_min) / \
            self.energy_bin_width if energy > 0 else 0
        cos_angle_nbins = round(
            (self.cosAngle_max - self.cosAngle_min) / self.cosAngle_bin_width)
        cos_angle_bin = (cos_angle - self.cosAngle_min) / \
            self.cosAngle_bin_width
        # guard against cosAngle = 1
        cos_angle_bin = min(cos_angle_bin, cos_angle_nbins - 1)
        if self.interpolation == 'linear':
            return self.pdf_linear(energy, cos_angle)
        elif self.interpolation == 'cubic':
            return self.pdf_cubic(energy, cos_angle)
        # no interpolation
        return self.pdf[int(energy_bin), int(cos_angle_bin)]


class ParametricPDF(PDF):
    #     energy_min: float
    #     energy_max: float
    #     energy_bin_width: float
    energy_binning: np.ndarray
    params: np.ndarray

    def eval(self, energy, cos_angle):
        #         if energy > self.energy_max:
        #             return 1.0
        #energy_bin = int((energy - self.energy_min) / self.energy_bin_width) if energy > 0 else 0
        energy_bin = np.searchsorted(self.energy_binning, energy)
        (g_1, g_2, sigma_1, sigma_2, c) = self.params[energy_bin]
        return g_1 * np.exp(- (cos_angle + 1) / sigma_1) + g_2 * np.exp((cos_angle - 1) / sigma_2) + c

    def __init__(self, name, energy_binning, params):
        super().__init__(name)
        # (self.energy_min, self.energy_max, self.energy_bin_width) = energy_binning
        self.energy_binning = np.asarray(energy_binning)
        self.params = params


def load_pdf_numeric(pdf_path, name=None):
    with open(pdf_path) as f:
        energy_binning = np.loadtxt(f, max_rows=1)
        cosAngle_binning = np.loadtxt(f, max_rows=1)
        pdf = np.loadtxt(f)
    # return energy_binning, cosAngle_binning, pdf
    return NumericPDF(name, energy_binning, cosAngle_binning, pdf)


def load_pdf_parameterized(pdf_path, name=None):
    with open(pdf_path) as f:
        energy_binning = np.loadtxt(f, max_rows=1)
        cosAngle_binning = np.loadtxt(f, max_rows=1)
        params = np.loadtxt(f)

    return ParametricPDF(name, energy_binning, params)

# %%


def charge_to_energy_clean(charge):
    m = 259.218
    b = 901.312
    return (charge - b) / m


def charge_to_energy_radio(charge):
    m = 248.111
    b = 2161.06
    return (charge - b) / m


def load_SN_file(filepath, with_radio=False):
    file = ROOT.TFile(filepath)
    tree = file.PointResTree.tr
    # nevts = tree.GetEntries()
    energies = []
    directions = []
    ievt = 0
    charge_to_energy = charge_to_energy_radio if with_radio else charge_to_energy_clean

    for event in tree:
        if event.NTrks == 0:
            continue
        energies.append(charge_to_energy(event.charge_corrected))

        directions.append([
            event.reco_e_dir.X(),
            event.reco_e_dir.Y(),
            event.reco_e_dir.Z()
        ])
        ievt += 1
    return np.asarray(energies), np.asarray(directions)


def get_truth_SN_dir(filepath):
    file = ROOT.TFile(filepath)
    tree = file.PointResTree.tr
    for event in tree:
        return np.array([event.truth_nu_dir.Theta(), event.truth_nu_dir.Phi()])


def sphere_to_xyz(direction):
    # [theta, phi] -> [x, y, z]
    return np.array([
        math.sin(direction[0]) * math.cos(direction[1]),
        math.sin(direction[0]) * math.sin(direction[1]),
        math.cos(direction[0])
    ])


def xyz_to_sphere(direction):
    # [x, y, z] => [theta, phi]
    # Identical to TVector3.Theta(), TVector3.Phi()
    direction_normalized = np.asarray(direction / np.linalg.norm(direction))
    phi = np.arctan2(direction_normalized[1], direction_normalized[0])
    perp = np.linalg.norm(direction_normalized[:2])
    theta = np.arctan2(perp, direction_normalized[2])
    return np.asarray([theta, phi])

# %%


class SupernovaPointing:
    """Class used to hold information about each supernova, and generate its loss function.
    """
    # Parallel list corresponding to each type of interaction
    PDFs: list[PDF]
    # tuple is [energy, direction]
    events_per_interaction: list[tuple[np.ndarray, np.ndarray]]
    weights: list[float]
    truth_dir: np.ndarray
    default_interpolation_type: str

    def __init__(self, PDFs, sn_event_files, weights=None, with_radio=False):
        """Class Constructor
        Args:
            @param PDFs: List of interaction information objects.
            @param sn_event_files: List of paths to ROOT files associated to each interaction.
            @param weights: Weight of each interaction. If None, they are all set to 1. Defaults to None.
        """
        self.PDFs = PDFs
        self.events_per_interaction = []
        for path in sn_event_files:
            self.events_per_interaction.append(
                load_SN_file(path, with_radio=with_radio))
        self.truth_dir = get_truth_SN_dir(sn_event_files[0])
        self.weights = [
            1.0] * len(self.events_per_interaction) if weights is None else weights

    def loss(self, sn_dir, weighting_factor=0, zero_bin=1e-4):
        """

        @param sn_dir: supernova direction, in form (theta, phi)
        @param weighting_factor: a linear weighting factor. Positive value biases higher energy bins. 0 means equal
        weighting.
        @param zero_bin: values to treat zero as (in preventing log(0)).
        @return: negative of log(likelihood)
        """
        sn_dir_xyz = sphere_to_xyz(sn_dir)
        loss = 0

        for pdf, events, weight in zip(self.PDFs, self.events_per_interaction, self.weights):
            interaction_loss = 0
            energies = events[0]
            cos_angles = events[1].dot(sn_dir_xyz)
            for energy, cos_angle in zip(energies, cos_angles):
                # # treat overly energetic events as having the same energy as PDF maxima
                if energy < 10:
                    continue
                pdf_value = pdf.eval(energy, cos_angle)
                # weighting formula is 1 + w * energy/100. Weight = 1 means 100MeV is doubly weighted
                pdf_value *= (1 + (energy/100) * weighting_factor)
                interaction_loss -= np.log(pdf_value)
            loss += interaction_loss * weight
        return loss

    def most_populated_dir(self, energy_cutoff=10):
        for events in self.events_per_interaction:
            directions = events[1][events[0] > energy_cutoff]
            width = 0.05
            bins = np.floor(directions/width)
            # print(mode(bins).mode)
            dir_xyz = mode(bins).mode.flatten()*width
            return xyz_to_sphere(dir_xyz)

    def high_energy_event_direction(self, votes=10):
        # Let the highest energy directions vote
        for events in self.events_per_interaction:
            idxs = np.argpartition(events[0], -votes)[-votes:]
            directions = []
            for idx in idxs:
                energy = events[0][idx]
                direction = events[1][idx]
                directions.append(direction)
            directions = np.asarray(directions)
            width = 0.25
            bins = np.floor(directions/width)
            dir_xyz = mode(bins).mode.flatten()*width
            return xyz_to_sphere(dir_xyz)

# %%


def error(pointer: SupernovaPointing, result, full_output=False, unit='deg'):
    assert unit in ['rad', 'deg'], "Unit must be deg or rad."
    if full_output:
        x0 = result[0]
    else:
        x0 = result

    angle = np.arccos(sphere_to_xyz(pointer.truth_dir).dot(sphere_to_xyz(x0)))
    return angle if unit == 'rad' else angle*180/np.pi


def details(pointer: SupernovaPointing, result: tuple):
    assert len(result) == 4, "Expect a 4-long tuple. Is this the full output?"
    # result [x0, fval, grid, jout]
    (x0, fval, grid, jout) = result
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(jout, extent=[-np.pi, np.pi, np.pi, 0],
               origin='upper', aspect='auto')
    plt.plot(pointer.truth_dir[1], pointer.truth_dir[0],
             'r*', label="True Direction")
    plt.plot(x0[1], x0[0], 'g*', label='Minimization Result')
    plt.colorbar()
    plt.legend()
    plt.title('Supernova Pointing Result')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')
    dot = sphere_to_xyz(pointer.truth_dir).dot(sphere_to_xyz(x0))

    # return degree deviation between minimization and truth
    return np.arccos(dot)*180/np.pi

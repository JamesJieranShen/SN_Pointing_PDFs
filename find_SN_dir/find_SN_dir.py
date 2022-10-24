# %%
from abc import ABCMeta, abstractmethod

from scipy.optimize import brute, optimize
from scipy.interpolate import interp2d
from scipy.stats import mode
from scipy.spatial.transform import Rotation
import ROOT
import math
from typing import Callable

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


# This file declares many utilities that can be used to reconstruct SN directions. In particular:
# - offers infrastructures to generated PDFs for each interaction channels.
# - utility functions to read PointResTree Root files.
# - A class for reconstructing SN direction, given PDFs and events.

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
    # energy_min: float
    # energy_max: float
    # energy_bin_width: float
    energy_binning: np.ndarray

    cosAngle_min: float
    cosAngle_max: float
    cosAngle_bin_width: float

    interpolation: str | None
    pdf: np.ndarray
    pdf_linear: Callable
    pdf_cubic: Callable

    def __init__(self, name, energy_binning, cosAngle_binning, pdf, interpolation=None):
        """
        Constructor
        @param name: name of the interaction
        @param energy_binning: bin edges for the energy bins. It is assumed that there is one overflow bin and one
        underflow bin.
        @param cosAngle_binning: (min, max, width) for cos(angle) binning.
        @param pdf: pdf data. If xscns = None, consider the entire input to be one PDF. If xscns is an array,
        consider this to be a list of pdfs, and mix according to the given cross-section ratio.
        @param interpolation: interpolation method. Can be None (no interpolation), 'linear', or 'cubic'.
        """
        super().__init__(name)
        # (self.energy_min, self.energy_max, self.energy_bin_width) = energy_binning
        self.energy_binning = energy_binning
        (self.cosAngle_min, self.cosAngle_max, self.cosAngle_bin_width) = cosAngle_binning
        self.pdf = np.array(pdf, copy=True)
        assert interpolation in (None, 'linear', 'cubic'), print(
            "Interpolation modes are None, linear, cubic")
        self.interpolation = interpolation
        energy_coords = np.arange(0, len(pdf))
        cos_angle_coords = np.arange(0, len(pdf[0]))

        if interpolation is not None:
            self.pdf_linear = interp2d(energy_coords, cos_angle_coords, np.transpose(pdf),
                                       bounds_error=False, fill_value=None, kind='linear')
            self.pdf_cubic = interp2d(energy_coords, cos_angle_coords, np.transpose(pdf),
                                      bounds_error=False, fill_value=None, kind='cubic')

    def eval(self, energy, cos_angle):
        # if energy > self.energy_max:
        #     return 1.0
        # energy_bin = (energy - self.energy_min) / \
        #     self.energy_bin_width if energy > 0 else 0
        energy_bin = np.searchsorted(self.energy_binning, energy)
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
        # energy_bin = int((energy - self.energy_min) / self.energy_bin_width) if energy > 0 else 0
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


# %% ROOT related methods

def load_SN_file(filepath, with_radio=False, return_nu_dir=False, show_progress=False):
    file = ROOT.TFile(filepath)
    tree = file.PointResTree.tr
    # nevts = tree.GetEntries()
    energies = []
    e_directions = []
    nu_directions = []
    ievt = 0
    charge_to_energy = charge_to_energy_radio if with_radio else charge_to_energy_clean
    iterable = tqdm(tree, total=tree.GetEntriesFast()) if show_progress else tree
    for event in iterable:
        if event.NTrks == 0:
            continue
        energies.append(charge_to_energy(event.charge_corrected))

        e_directions.append([
            event.reco_e_dir.X(),
            event.reco_e_dir.Y(),
            event.reco_e_dir.Z()
        ])
        if return_nu_dir:
            nu_directions.append([
                event.truth_nu_dir.X(),
                event.truth_nu_dir.Y(),
                event.truth_nu_dir.Z()
            ])
        ievt += 1
    if return_nu_dir:
        return np.asarray(energies), np.asarray(e_directions), np.asarray(nu_directions)
    return np.asarray(energies), np.asarray(e_directions)


def get_truth_SN_dir(filepath):
    file = ROOT.TFile(filepath)
    tree = file.PointResTree.tr
    for event in tree:
        return np.array([event.truth_nu_dir.Theta(), event.truth_nu_dir.Phi()])


# %% Other Utility Functions
def charge_to_energy_clean(charge):
    m = 259.218
    b = 901.312
    return (charge - b) / m


def charge_to_energy_radio(charge):
    m = 248.111
    b = 2161.06
    return (charge - b) / m


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
    return np.array([theta, phi])


# %% SN_Pointer Class and its helpers
def get_expected_counts(xscns, confusion_matrix, channel_id_list=None):
    """
    Generate expected counts for selected pointing channels.
    @param channel_id_list: index of the interactions that we generate weights for. If None, assume all channels are
    used.
    @param xscns: total numbers of expected events for each interaction.
    @param confusion_matrix: Confusion matrix of the interactions. element [i, j] represent the number of events in
    interaction i that is identified as character j. Each row of the confusion matrix should be normalized (divided
    by their total cross-section). However, the sum might not be zero, if detection efficiency is not perfect.
    @return: A matrix of expected counts. expected_counts[i][j] is the expected number of
            events in interaction j that is present in channel i.
    """
    total_expected_count = (np.diag(xscns) @ confusion_matrix).T
    if channel_id_list is None:
        return total_expected_count
    return np.array(total_expected_count[np.asarray(channel_id_list)])

def pre_rotate_events(orig_event_file, standard_dir=np.array([0, 0, 1])):
    """
    Rotate all events in `orig_event_file` to have the same neutrino direction.
    @param standard_dir: The neutrino direction of all output events.
    @return: The same event format, but with rotated directions.
    """
    energies, e_dir, nu_dir = orig_event_file
    sn_ortho = np.cross(nu_dir, standard_dir)
    desired_frame = [[standard_dir, orth] for orth in sn_ortho]
    raw_frame = [[nu_dir[i], sn_ortho[i]] for i in range(len(sn_ortho))]

    rotated_e_dir = np.empty(e_dir.shape)
    for idx, (raw, desired) in tqdm(enumerate(zip(raw_frame, desired_frame)), total=len(raw_frame)):
        (rot, _) = Rotation.align_vectors(desired, raw)
        rotated_e_dir[idx] = rot.apply(e_dir[idx])
    return [energies, rotated_e_dir, np.array([standard_dir]*len(energies))]

def draw_events(sn_event_files, event_counts, truth_dir, rng=None, pre_rotated=False):
    """
    Draw supernova events from the event files, rotate them to the correct directions.
    @param sn_event_files: (energy, directions) tuple for the events
    @param event_counts: number of events to draw from each root file
    @param truth_dir: the direction of the neutrino to align with
    @param rng: random number generator. If None, use `rng.random.default_rng()`
    @param pre_rotated: If True, assume input events are pre-rotated to the same direction.
    """
    if rng is None:
        rng = np.random.default_rng()
    truth_dir_xyz = sphere_to_xyz(truth_dir)
    energies = []
    directions = []
    for event, count in zip(sn_event_files, event_counts):
        if count <= 0:
            continue
        file_energies, file_e_directions, file_nu_directions = event
        nevts = file_energies.size
        if pre_rotated:
            standard_sn_dir = file_nu_directions[0] # Assume every single event in file has the same direction.and
            ortho = np.cross(standard_sn_dir, truth_dir_xyz)
            desired_frame = [truth_dir_xyz, ortho]
            raw_frame = [standard_sn_dir, ortho]
            (standard_rotation, rmsd) = Rotation.align_vectors(desired_frame, raw_frame)
        for _ in range(count):
            idx = rng.choice(nevts)
            energies.append(file_energies[idx])
            raw_e_dir = file_e_directions[idx]
            raw_sn_dir = file_nu_directions[idx]
            # do rotation
            # define a rotation. Rotation is ambiguous since we only need to match one vector. Eliminate ambiguity by
            # assuming that the direction that is orthogonal to the two SN directions is the rotational axis.
            if pre_rotated:
                rotated_e_dir = standard_rotation.apply(raw_e_dir)
            else:
                sn_orthogonal = np.cross(raw_sn_dir, truth_dir_xyz)  # does not need to be normalized
                desired_frame = [truth_dir_xyz, sn_orthogonal]
                raw_frame = [raw_sn_dir, sn_orthogonal]
                (rotation, rmsd) = Rotation.align_vectors(desired_frame, raw_frame)  # generate rotation raw -> desired
                rotated_e_dir = rotation.apply(raw_e_dir)
            directions.append(rotated_e_dir)
    return np.array(energies), np.array(directions)


class SupernovaPointing:
    """Class used to hold information about each supernova, and generate its loss function.
    The class takes truth files for different interactions, and conducts maximum likelihood fitting to reconstructed
    channels. The number of events from each interaction to each channel is described using a matrix of expected counts.
    """
    # Parallel list corresponding to each type of interaction
    PDFs: list[PDF]
    # tuple is [energy, direction]
    events_per_channel: list[tuple[np.ndarray, np.ndarray]]
    channel_weights: list[float]
    truth_dir: np.ndarray
    expected_counts_normalized: np.ndarray
    is_pure: bool
    default_interpolation_type: str

    def __init__(self, PDFs, sn_event_files,
                 synthetic=False, expected_counts=None,
                 poisson_count=False, channel_weights=None, with_radio=False, sn_dir=None, pre_rotated_files=False):
        """Class Constructor
        Args:
            @param PDFs: List of interaction information objects. The number of PDFs must be the same as the number
                        of sn_event_files.
            @param sn_event_files: If synthetic is set to false, list of paths to ROOT files associated to each
                    interaction. all events in the listed files will be considered part of the supernova event.
                    Otherwise, this is a list of tuples (energy, direction).
                    events will be drawn from each file according to synthetic_counts.
            @param synthetic: If True, synthesize supernova events by drawing events from a large pool of events. If
            false, all events will be used.
            @param expected_counts: A matrix of expected counts. expected_counts[i][j] is the expected number of
            events in interaction j that is present in channel i. In the non-synthetic case, this matrix is
            considered to be diagonal (all events are pure).
            @param poisson_count: If True, interpret synthetic counts as expected values in a poisson distribution.
            @param channel_weights: Weight of each channel. If None, they are all set to 1. Defaults to None.
            @param sn_dir: set a fixed supernova direction. Default to None (randomly select direction).
            @param pre_rotated_files: If True, assume all input sn_event_files are pre-rotated to have nu direction of
            (0, 0, 1). This optimizes the event rotation during draw_events.
        """
        if synthetic:
            if channel_weights is None:
                channel_weights = [1.0] * len(expected_counts)
            assert (len(PDFs) == len(sn_event_files) == len(expected_counts[0])) \
                and (len(channel_weights) == len(expected_counts)), \
                "Inputs do not agree on correct number of channels and interactions"
        if not synthetic:
            channel_weights = [1.0] * len(PDFs)
            expected_counts = np.identity(len(PDFs))
        self.is_pure = synthetic  # Could also check if synthetic cases are diagonal
        self.PDFs = PDFs
        self.events_per_channel = []
        self.channel_weights = channel_weights
        self.expected_counts_normalized = np.array(expected_counts, copy=True, dtype=float)

        # Use generated SN events
        if not synthetic:
            self.truth_dir = get_truth_SN_dir(sn_event_files[0])
            for path in sn_event_files:
                self.events_per_channel.append(
                    load_SN_file(path, with_radio=with_radio))
        else:  # synthesize SN events from a large pool of data (from the PDF root files, for example)
            rng = np.random.default_rng()
            # Randomly select a truth direction.
            self.truth_dir = np.array([rng.uniform(0, np.pi), rng.uniform(-np.pi, np.pi)]) if sn_dir is None else np.array(sn_dir)
            if poisson_count:
                expected_counts = rng.poisson(expected_counts)
            for channel_id, counts in enumerate(expected_counts):
                self.events_per_channel.append(
                    draw_events(sn_event_files, np.rint(counts).astype(int), self.truth_dir, rng=rng, pre_rotated=pre_rotated_files))
                self.expected_counts_normalized[channel_id] /= np.sum(counts)

    def loss(self, sn_dir, zero_bin=1e-4):
        """

        @param sn_dir: guessed supernova direction, in form (theta, phi)
        @param zero_bin: values to treat zero as (in preventing log(0)).
        @return: negative of log(likelihood)
        """
        sn_dir_xyz = sphere_to_xyz(sn_dir)
        loss = 0

        for channel_id, (events, channel_weight, interaction_weights) in \
                enumerate(zip(self.events_per_channel, self.channel_weights, self.expected_counts_normalized)):
            channel_loss = 0
            energies = events[0]
            cos_angles = events[1].dot(sn_dir_xyz)
            for energy, cos_angle in zip(energies, cos_angles):
                if energy < 10:
                    continue
                pdf_value = 0.0
                if self.is_pure:
                    # If pure, channel is the same thing as interactions
                    pdf = self.PDFs[channel_id]
                    pdf_value = pdf.eval(energy, cos_angle)
                else:  # mixed events
                    for i, pdf in enumerate(self.PDFs):
                        pdf_value += pdf.eval(energy, cos_angle) * interaction_weights[i]

                pdf_value = max(pdf_value, zero_bin)
                channel_loss -= np.log(pdf_value)
            loss += channel_loss * channel_weight
        return loss

    def most_populated_dir(self, energy_cutoff=10):
        for events in self.events_per_channel:
            directions = events[1][events[0] > energy_cutoff]
            width = 0.05
            bins = np.floor(directions / width)
            # print(mode(bins).mode)
            dir_xyz = mode(bins).mode.flatten() * width
            return xyz_to_sphere(dir_xyz)

    def high_energy_event_direction(self, votes=10):
        # Let the highest energy directions vote
        for events in self.events_per_channel:
            idxs = np.argpartition(events[0], -votes)[-votes:]
            directions = []
            for idx in idxs:
                energy = events[0][idx]
                direction = events[1][idx]
                directions.append(direction)
            directions = np.asarray(directions)
            width = 0.25
            bins = np.floor(directions / width)
            dir_xyz = mode(bins).mode.flatten() * width
            return xyz_to_sphere(dir_xyz)


# %% Monitoring/Visualization methods

def error(pointer: SupernovaPointing, result, full_output=False, unit='deg'):
    assert unit in ['rad', 'deg'], "Unit must be deg or rad."
    if full_output:
        x0 = result[0]
    else:
        x0 = result

    angle = np.arccos(sphere_to_xyz(pointer.truth_dir).dot(sphere_to_xyz(x0)))
    return angle if unit == 'rad' else angle * 180 / np.pi


def details(pointer: SupernovaPointing, result: tuple):
    assert len(result) == 4, "Expect a 4-long tuple. Is this the full output?"
    # result [x0, fval, grid, jout]
#     (x0, fval, grid, jout) = result
#     plt.figure(figsize=(8, 6), dpi=120)
#     plt.imshow(jout, extent=[-np.pi, np.pi, np.pi, 0],
#                origin='upper', aspect='auto')
#     plt.plot(pointer.truth_dir[1], pointer.truth_dir[0],
#              'r*', label="True Direction")
#     plt.plot(x0[1], x0[0], 'g*', label='Minimization Result')
#     plt.colorbar()
#     plt.legend()
#     plt.title('Supernova Pointing Result')
#     plt.xlabel(r'$\phi$')
#     plt.ylabel(r'$\theta$')
#     dot = sphere_to_xyz(pointer.truth_dir).dot(sphere_to_xyz(x0))
#     plt.show()
#     # return degree deviation between minimization and truth
#     return np.arccos(dot) * 180 / np.pi

    NSIDE = 16
    NPIX = hp.nside2npix(NSIDE)

    (x0, fval, grid, jout) = result
    (theta, phi) = grid
    theta = theta.flatten()
    phi = phi.flatten()
    jout = jout.flatten()

    idx = hp.ang2pix(NSIDE, theta, phi)
    pixels = np.zeros(NPIX)
    pixels[idx] = jout
    # pixels = hp.smoothing(pixels, fwhm=np.radians(0.))
    hp.mollview(pixels, title="Skymap of Neutrino direction", norm='hist')
    hp.projplot(pointer.truth_dir, 'r*', label="Truth SN Direction")
    hp.projplot(x0, 'g*', label='Minimization Result')
    plt.legend()
    hp.graticule()
    dot = sphere_to_xyz(pointer.truth_dir).dot(sphere_to_xyz(x0))
    return np.arccos(dot) * 180 / np.pi


#!/usr/bin/python
# This script takes the file path of a PointResTree root file, and a parameterized PDF.
# The output file has the name *_pdf_param.dat, and is placed in the current working directory.
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

import ROOT
import numpy as np
import sys
import getopt
import os
from tqdm import tqdm, trange

# Custom binning
# AJ's binning + higher energy bins
# list(range(10, 80, 10))
energy_bins = list(range(2, 40, 2)) + [40, 50, 60, 70] 

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'o:', ['clean', 'radio'])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    assert (len(args) == 1), print_help()
    # print(opts)
    file_path = args[0]
    assert os.path.exists(file_path), "ROOTFile does not exist!"
    file_name = os.path.basename(file_path)
    print(f"Loading {file_path}.")

    # replace ROOT extension with _pdf.dat. filename.root -> filename_pdf.dat
    pdf_filename = ''.join(os.path.splitext(file_name)[:-1]) + '_pdf_param.dat'
    has_radio = False
    for opt in opts:
        if '--radio' in opt:
            has_radio = True
        if '--clean' in opt:
            has_radio = False
        if '-o' in opt:
            # print(opt)
            pdf_filename = opt[-1] + '/' + pdf_filename
    print(
        f"Assuming ROOTFile has {'no ' if not has_radio else ''}radiologicals")

    file = ROOT.TFile(file_path)
    tree = file.PointResTree.tr
    if has_radio:
        m = 248.111
        b = 2161.06
    else:
        m = 259.218
        b = 901.312
    def charge_to_energy(q): return (q-b)/m

    functype = 'exp' # exponential distribution
    # e_binwidth = 10
    # e_min = 0
    # e_max = 70
    nbin_en = len(energy_bins) + 1
    nbin_cosangle = 100
    cosangle_binwidth = 2 / nbin_cosangle
    pdf = np.zeros([nbin_en, nbin_cosangle])
    for e in tqdm(tree, total=tree.GetEntries()):
        if e.NTrks == 0:
            continue
        en = charge_to_energy(e.charge_corrected)
        cosAngle = e.truth_nu_dir.Dot(e.reco_e_dir)
        en_binidx = np.searchsorted(energy_bins, en)
        cosangle_binidx = int((cosAngle + 1) // cosangle_binwidth)
        pdf[en_binidx, cosangle_binidx] += 1
    # normalize pdf for each energy bin
    for i in range(len(pdf)):
        # normalize first to improve fit performance. Normalized again after the parameters are determined.
        pdf[i] = pdf[i] / np.sum(pdf[i])
    # if np.min(pdf) <= 0:
    #     print("WARNING: PDF has zero bins. This may result in log(0) down the line...", file=sys.stderr)
    # np.where(pdf==0, pdf, 0.0001)
    # Do the fit
    fit_x = np.linspace(-1, 1, num=nbin_cosangle) + 2/nbin_cosangle/2 # X axis of fit
    NParams = 5
    params = np.zeros((nbin_en, NParams))
    for (i, pdf_slice) in enumerate(pdf):
        params[i] = fit_bipeak(fit_x, pdf_slice)

    # Format: 
    # Parameterized. Generated from ...
    # Energy min, Energy max, Energy binwidth
    # CosAngle min, CosAngle max
    # functype
    # Main text

    print(f"Writing to {pdf_filename}...")
    # print(params)
    with open(pdf_filename, 'w') as f:
        f.write(f'# From {file_name}\n')
        f.write(f'# Parameterized {functype}\n')
        # f.write('# Energy min, Energy max\n')
        # f.write(f'{e_min} {e_max}\n')
        f.write(f'# Energy Bin Left Edges (there is one underflow and overflow bin)\n')
        np.savetxt(f, [energy_bins], fmt='%.4e')
        f.write('# CosAngle min, CosAngle max\n')
        f.write('-1 1\n\n')
        np.savetxt(f, params)

# fit function: 
def bi_peak_distribution(x, g1, g2, sigma_1, sigma_2, c):
    return g1 * np.exp(- (x+1)/sigma_1) + g2 * np.exp((x-1)/sigma_2)+c

# integral from -1 to 1
def integral(g1, g2, sigma_1, sigma_2, c):
    return 2*c + g1 * (1 - np.exp(-2/sigma_1))*sigma_1 + g2 * (1 - np.exp(-2/sigma_2))* sigma_2

# noramzlie function so that integral = 1
def normalize(params):
    integral_result = integral(*params)
    return np.asarray(params) * np.array([1/integral_result, 1/integral_result, 1, 1, 1/integral_result])


def fit_bipeak(bin_right_edge, pdf_slice):
    def bi_peak_distribution(x, g1, g2, sigma_1, sigma_2, c):
        return g1 * np.exp(- (x+1)/sigma_1) + g2 * np.exp((x-1)/sigma_2)+c
    
    p0 = np.array([1, 1, 1, 1, 0]) # initial condition
    bounds = (np.zeros(p0.shape), np.array([np.inf, np.inf, np.inf, np.inf, 0.1]))
    (params, cov) = curve_fit(bi_peak_distribution, bin_right_edge, pdf_slice, p0, bounds=bounds)
    rms = np.mean((pdf_slice - bi_peak_distribution(bin_right_edge, *params))**2)
    # params = normalize(params)
    if rms > 0.1:
        print("WARNING: LARGE RMS Detected. Current RMS is {rms :.2e}")
    return params


def print_help():
    print("make_PDF_parameterized.py [-o outdir] [--radio|--clean] <filename>")


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/python
# This script takes the file path of a PointResTree root file, and generate PDF accordingly.
# The output file has the name *_pdf.dat, and is placed in the current working directory.
import ROOT
import numpy as np
import sys
import getopt
import os
from tqdm import tqdm, trange
from scipy.ndimage import gaussian_filter1d

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
    pdf_filename = ''.join(os.path.splitext(file_name)[:-1]) + '_pdf.dat'
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

    e_binwidth = 10
    e_min = 0
    e_max = 70
    nbin_en = int((e_max - e_min)/e_binwidth)
    nbin_cosangle = 100
    cosangle_binwidth = 2 / nbin_cosangle
    pdf = np.zeros([nbin_en, nbin_cosangle])
    for e in tqdm(tree, total=tree.GetEntries()):
        if e.NTrks == 0:
            continue
        en = charge_to_energy(e.charge_corrected)
        if en > e_max:
            continue
        cosAngle = e.truth_nu_dir.Dot(e.reco_e_dir)
        en_binidx = int((en-e_min)//e_binwidth) if en > 0 else 0
        cosangle_binidx = int((cosAngle + 1) // cosangle_binwidth)
        pdf[en_binidx, cosangle_binidx] += 1
    # normalize pdf for each energy bin
    for i in range(len(pdf)):
        pdf[i] = gaussian_filter1d(pdf[i], 1, order=0, mode='nearest')
        pdf[i] = pdf[i] / np.sum(pdf[i])
    # if np.min(pdf) <= 0:
    #     print("WARNING: PDF has zero bins. This may result in log(0) down the line...", file=sys.stderr)
    pdf = np.where(pdf==0, 0.0001, pdf)
    print(f"Writing to {pdf_filename}...")
    with open(pdf_filename, 'w') as f:
        f.write(f'# Generated from {file_name}\n')
        f.write('# Energy min, Energy max, Energy binwidth\n')
        f.write(f'{e_min} {e_max} {e_binwidth}\n')
        f.write('# CosAngle min, CosAngle max, CosAngle binwidth\n')
        f.write(f'-1 1 {cosangle_binwidth}\n\n')
        np.savetxt(f, pdf)


def print_help():
    print("make_PDF.py [-o outdir] [--radio|--clean] <filename>")


if __name__ == "__main__":
    main(sys.argv[1:])

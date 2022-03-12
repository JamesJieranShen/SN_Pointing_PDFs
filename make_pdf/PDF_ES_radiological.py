#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import ROOT
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


# In[2]:


# read in file
import os
file_name = f"{os.environ.get('HEP')}/data/ES_gvkm/ES_gvkm_radiological_0201.root"
# file_name = f"{os.environ.get('HEP')}/data/marley_gvkm_clean_0114.root"
file = ROOT.TFile(file_name)
tree=file.PointResTree.tr


# In[3]:


# loop through file
# define linear coefficients for energy reconstruction
def charge_to_energy_clean(charge):
    m = 248.111
    b = 2161.06
    return (charge - b)/m
cosAngle_list = []
sample_size_per_bin = []
nbin_en = 7
e_max = 70
nbin_cosangle = 20
pdf = np.zeros([nbin_en, nbin_cosangle])
for e in tqdm(tree, total=tree.GetEntries()):
    if e.NTrks==0: continue
    en = charge_to_energy_clean(e.charge_corrected)
    if en > e_max: continue
    cosAngle = e.truth_nu_dir.Dot(e.reco_e_dir)
    en_binidx = int(en//(e_max/nbin_en)) if en > 0 else 0
    cosangle_binidx = int((cosAngle+1)//(2/nbin_cosangle))
    pdf[en_binidx, cosangle_binidx]+=1
# normalize pdf for each energy bin
for i in range(len(pdf)):
    energy_slice = pdf[i]
    sample_size_per_bin.append(np.sum(pdf[i]))
    pdf[i] = pdf[i] / np.sum(pdf[i])
# np.savetxt('ES_gvkm_radiological.dat', pdf)


# In[4]:


# fig, ax = plt.subplots(figsize=(6,6))
# 
# ax.imshow(pdf, extent=[-1,1,0,e_max], aspect='auto')
# ax.set_title("PDF for Elastic Scattering Events")
# ax.set_xlabel(r'$\cos(\theta)$')
# ax.set_ylabel(r'$e^-$ energy (MeV)')


# In[5]:


# energy slices
cosAngle_bins = np.linspace(-1, 1, num=20, endpoint=False) + 2/20/2
# print(cosAngle_bins)
# for i, (pdf_slice, sample_size) in enumerate(zip(pdf, sample_size_per_bin)):
#     plt.figure()
#     plt.title(f"Cosine Scattering Angle Distribution, Observed Energy {i*5} - {(i+1)*5} MeV\n"            f"Sample Size: {sample_size}")
#     plt.plot(cosAngle_bins, pdf_slice)


# In[7]:


def bi_peak_distribution(x, g1, g2, sigma_1, sigma_2):
    return g1 * np.exp(-sigma_1 * (x+1)) # + g2 * np.exp(sigma_2 * (x-1))
data = pdf[1]

from scipy.optimize import curve_fit
slice_fit = pdf[1]
print(slice_fit)
# res = curve_fit(bi_peak_distribution, cosAngle_bins, slice_fit, [0.1, 0.4, 5, 10], bounds=([0, 0, 0.5, 0.5], [1, 1, 20, 20]))
# print(res)
# x = np.linspace(-1, 1, 100)
# plt.plot(cosAngle_bins, pdf[1])
# plt.plot(x, bi_peak_distribution(x, 0.13, 0.4, 5, 10))


# In[ ]:


res = curve_fit(bi_peak_distribution, cosAngle_bins, slice_fit, [0.1, 0.4, 5, 10], bounds=([0.01, 0.01, 0.5, 0.5], [1, 1, 20, 20]), method='dogbox')
print(res)


#!/usr/bin/env python

import SamplePdf
import numpy as np
import os
from numpy.random import uniform
import pickle as pkl

num_samples = 5000

uni_samples = np.zeros(num_samples)

# TODO: draw uniform samples
uni_samples = uniform(size=num_samples)

# create instance of our new PDF sampler
my_pdf_sampler = SamplePdf.SamplePdf(num_samples)

# feed the uniform samples and create our custom ones
# TODO this function needs to be implemented in SamplePdf.py
new_samples = my_pdf_sampler.sample_pdf(uni_samples)

my_pdf_sampler.plot_result()

# safe the result in a struct
pkl.dump(new_samples, open("results.pkl", 'wb'))
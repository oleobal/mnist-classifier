#!/usr/bin/python3
"""
Minimum distance classifier, with PCA (principal component analysis)
"""
# pip/conda package scikit-learn
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
import sklearn


import numpy as np
import pdb
import pprint
pp = pprint.PrettyPrinter(indent=4)		



"""
Learning from labeled set
"""
trndata = np.load("data/trn_img.npy")
trnlabl = np.load("data/trn_lbl.npy")


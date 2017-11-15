#!/usr/bin/python3

import numpy as np
import pprint

data = np.load("data/trn_img.npy")
label = np.load("data/trn_lbl.npy")

# data, sliced up by label
dataBrrm = [[], [], [], [], [], [], [], [], [], []]

# average vector for each label
app = [[], [], [], [], [], [], [], [], [], []]

for i in range(len(data)):
	dataBrrm[label[i]].append(data[i])

for i in range(10):
	for j in range(784):
		app[i].append(np.mean(dataBrrm[i][j]))

pp = pprint.PrettyPrinter(indent=4)		
pp.pprint(app)

"""
for i in range(len(data)):
	
	import matplotlib.pyplot as plt
	img = data[i].reshape(28,28)
	plt.imshow(img, plt.cm.gray)
	plt.show()
	
	print(label[i])
"""

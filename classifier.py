#!/usr/bin/python3
"""
Minimum distance classifier
"""

from utility import *
import numpy as np
from sys import argv

import pdb
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)		




optAverageImages = "-a"
optPCA = "-p"
optPCAcomps = 10

if "help" in argv or "-h" in argv:
	print("Minimum distance classifier")
	print("Options :")
	print(optAverageImages+" : display average images once training is done")
	print(optPCA+" <n> : use PCA (principal component analysis), with <n> components")
	exit(0)


"""
Learning from labeled set
"""

print("Starting reading & training..")

data = np.load("data/trn_img.npy")
label = np.load("data/trn_lbl.npy")

#pp.pprint(label)

# data, sliced up by label
dataSliced = [[], [], [], [], [], [], [], [], [], []]

# average vectors for each label
app = [[], [], [], [], [], [], [], [], [], []]


for i in range(len(data)):
	dataSliced[label[i]].append(data[i])

for i in range(10): #which category
	"""
	for j in dataSliced[i]: #which image
		
		px = 0
		for z in range(784): #which pixel
			px = np.mean(dataSliced[i][j], z)
	"""
	for j in range(784):
		app[i]=np.mean(dataSliced[i], axis=0)#wot
		print(progressBar(i*784+j,7840,29), end="\r")
		
print(" "*29, end="\r")# erase progress line

if optPCA in argv :
	print("PCA  processing in progress..")
	# pip/conda package scikit-learn
	# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
	from sklearn.decomposition import PCA
	optPCAcomps = int(argv[argv.index(optPCA)+1])
	pca = PCA(n_components=optPCAcomps)
	for i in range(10):
		app[i] = pca.fit(np.array(app[i]).reshape(28,28)).components_
		app[i] = np.hstack(app[i])



if optAverageImages in argv:
	# print average images
	import matplotlib.pyplot as plt
	import matplotlib.image as mpimg
	fig = plt.figure()

	for i in range(len(app)):
		z = fig.add_subplot(2,5,i+1)
		z.set_title(str(i))
		plt.imshow(np.array(app[i]).reshape(optPCAcomps,28), plt.cm.gray)
	plt.show()

	
"""
averageVectors : an array of arrays (average images)
vector : an array (image)

returns : the index of the corresponding closest average
"""
def getMinDistanceIndex(averageVectors, vector):
	distance = [0,0,0,0,0,0,0,0,0,0]
	for d in range(10):
		for i in range(len(vector)):
			distance[d] += np.absolute(averageVectors[d][i]-vector[i])
	return distance.index(min(distance))

"""
Evaluating on test data
"""
print("Finished training, analysis..")


testdata = np.load("data/tst_img.npy")
testlabel = np.load("data/tst_lbl.npy")

total = [0,0,0,0,0,0,0,0,0,0]
nbWrong = [0,0,0,0,0,0,0,0,0,0]
guesses = [0,0,0,0,0,0,0,0,0,0]

for i in range(len(testdata)) :
	# compute average and see to which it is closest
	#v = np.mean(testdata[i])
	#guess = np.abs(app - v).argmin()
	guess = getMinDistanceIndex(app, testdata[i])
	if (guess != testlabel[i]):
		nbWrong[testlabel[i]]+=1
	#print("guess/actual : "+str(guess)+"/"+str(testlabel[i]))
	print(progressBar(i,len(testdata),29), end="\r")
	total[testlabel[i]]+=1
	guesses[guess]+=1

print(" "*29, end="\r")# erase progress line

guessTot = sum(guesses)
print("Approx guesses distribution :")
print("0  1  2  3  4  5  6  7  8  9")
# total très possiblement != 100 à cause de l'arrondi
for i in range(10):
	prop = round((guesses[i]/guessTot)*100)
	if prop < 10:
		prop = str(prop)+"% "
	elif prop < 100:
		prop = str(prop)+"%"
	else:
		prop = str(prop)
	print(prop, end="");
	
print("\n-----------------------------")


bigWrong=0
bigTotal=0
for i in range(10):
	print("Cat. "+str(i)+" failure rate  : "+getNicePercent((nbWrong[i]/total[i])*100))
	bigWrong+=nbWrong[i]
	bigTotal+=total[i]
print("-----------------------------")
print("Total  failure rate  : "+getNicePercent((bigWrong/bigTotal)*100))

#TODO confusion matrix

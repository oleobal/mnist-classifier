#!/usr/bin/python3
"""
Minimum distance classifier
"""

from utility import *
import numpy as np
from sys import argv

#import pdb
#from pprint import PrettyPrinter
#pp = PrettyPrinter(indent=4)		




optAverageImages = "-a"
optPCA = "-p"
optPCAcomps = 28
displayWidth=39

if "help" in argv or "-h" in argv:
	print("Minimum distance classifier")
	print("Options :")
	print(optAverageImages+" : display average images once training is done")
	print(optPCA+" <n> : use PCA (principal component analysis), with <n> components")
	exit(0)


"""
Learning from labeled set
"""

print("Reading & training..")

data_unproc = np.load("data/trn_img.npy")
label = np.load("data/trn_lbl.npy")

#pp.pprint(label)

# data, sliced up by label
dataSliced = [[], [], [], [], [], [], [], [], [], []]

# average vectors for each label
app = [[], [], [], [], [], [], [], [], [], []]


totalLearningOps = 0
progress = 0

if optPCA in argv :
	print("(PCA processing enabled)")
	# pip/conda package scikit-learn
	# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
	from sklearn.decomposition import PCA
	optPCAcomps = int(argv[argv.index(optPCA)+1])
	data = []
	totalLearningOps+=len(data_unproc)
	totalLearningOps+=optPCAcomps*28*10
	pca = PCA(n_components=optPCAcomps)
	for i in range(len(data_unproc)):
		data.append(np.hstack(pca.fit(np.array(data_unproc[i]).reshape(28,28)).components_))
		progress+=1
		print(progressBar(progress,totalLearningOps,displayWidth), end="\r")
else:
	totalLearningOps+=28*28*10
	data = data_unproc

# not counted in progress bar because too fast
for i in range(len(data)):
	dataSliced[label[i]].append(data[i])

for i in range(10): #which category
	for j in range(len(data[0])):
		app[i]=np.mean(dataSliced[i], axis=0)#wot
		progress+=1
		print(progressBar(progress,totalLearningOps,displayWidth), end="\r")
		
print(" "*displayWidth, end="\r")# erase progress line



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


testdata_unproc = np.load("data/tst_img.npy")
testlabel = np.load("data/tst_lbl.npy")


if optPCA in argv :
	testdata = []
	optPCAcomps = int(argv[argv.index(optPCA)+1])
	for i in range(len(testdata_unproc)):
		testdata.append(np.hstack(pca.fit(np.array(testdata_unproc[i]).reshape(28,28)).components_))
else:
	testdata = testdata_unproc

total = [0,0,0,0,0,0,0,0,0,0]
nbWrong = [0,0,0,0,0,0,0,0,0,0]
guesses = [0,0,0,0,0,0,0,0,0,0]

# +- label
# |
# guess
confusionMatrix =[
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0]
]

for i in range(len(testdata)) :
	# compute average and see to which it is closest
	guess = getMinDistanceIndex(app, testdata[i])
	if (guess != testlabel[i]):
		nbWrong[testlabel[i]]+=1
	print(progressBar(i,len(testdata),displayWidth), end="\r")
	total[testlabel[i]]+=1
	confusionMatrix[guess][testlabel[i]]+=1
	guesses[guess]+=1 # it can be computed from the confusion matrix, but honestly, the computer is here to do it for us

print(" "*displayWidth, end="\r")# erase progress line


print("Approximate guesses distribution :")
print("0- -1- -2- -3- -4- -5- -6- -7- -8- -9")
# total très possiblement != 100 à cause de l'arrondi
guessTot = sum(guesses)
for i in range(10):
	prop = round((guesses[i]/guessTot)*100)
	if prop < 10:
		prop = str(prop)+"   "
	elif prop < 100:
		prop = str(prop)+"  "
	else:
		prop = str(prop)+" "
	print(prop, end="");

print("\n"+"-"*displayWidth)

print("Confusion matrix (x:label, y:guess) :")

print("0- -1- -2- -3- -4- -5- -6- -7- -8- -9")
for i in range(10):
	for j in range(10):
		if (confusionMatrix[i][j] < 10):
			print(confusionMatrix[i][j], end="   ")
		elif confusionMatrix[i][j] < 100:
			print(confusionMatrix[i][j], end="  ")
		else:
			print(confusionMatrix[i][j], end=" ")
	print()

print("\n"+"-"*displayWidth)

print("Failure rate per category :")
print("0- -1- -2- -3- -4- -5- -6- -7- -8- -9")
bigWrong=0
bigTotal=0
for i in range(10):
	print(getNiceRound((nbWrong[i]/total[i])*100), end="");
	bigWrong+=nbWrong[i]
	bigTotal+=total[i]
print("\n"+"-"*displayWidth)
print("Total  failure rate  : "+getNicePercent((bigWrong/bigTotal)*100,2))

#TODO confusion matrix
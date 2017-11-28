#!/usr/bin/python3

import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)		



"""
Learning from labeled set

"""
data = np.load("data/trn_img.npy")
label = np.load("data/trn_lbl.npy")

#pp.pprint(label)
"""
for i in range(len(data)):
	
	import matplotlib.pyplot as plt
	img = data[i].reshape(28,28)
	plt.imshow(img, plt.cm.gray)
	plt.show()
	
	print(label[i])
"""

# data, sliced up by label
dataSliced = [[], [], [], [], [], [], [], [], [], []]

# average vectors for each label
app = [[], [], [], [], [], [], [], [], [], []]

for i in range(len(data)):
	dataSliced[label[i]].append(data[i])

for i in range(10):
	for j in range(784):
		app[i].append(np.mean(dataSliced[i][j]))
		
"""
# here we have the average vector for each picture
# so we average again (so it's for each label)
for i in range(10):
	app[i] = np.mean(app[i])
"""	
#pp.pprint(app)


"""
averageVectors : an array of arrays (average images)
vector : an array (image)

returns : the index of the corresponding closest average
"""
def getMinDistanceIndex(averageVectors, vector):
	distance = [0,0,0,0,0,0,0,0,0,0]
	for d in range(10):
		buf=0
		for i in range(len(vector)):
			buf += np.abs(averageVectors[d][i]-vector[i])
		distance[d] = buf
	return distance.index(min(distance))


def progressBar(i, maxi, length):
	nbchars = round((i/maxi)*(length-5))
	result = "["+nbchars*"="+(length-5-nbchars)*"-"+"]"
	pcent = round((i/maxi)*100)
	if pcent<10:
		result+="0"+str(pcent)+"%"
	elif pcent<100:
		result+=str(pcent)+"%"
	else:
		result+=str(pcent)
	return result

"""
Evaluating on test data
"""



testdata = np.load("data/tst_img.npy")
testlabel = np.load("data/tst_lbl.npy")

total = [0,0,0,0,0,0,0,0,0,0]
nbWrong = [0,0,0,0,0,0,0,0,0,0]

for i in range(len(testdata)) :
	# compute average and see to which it is closest
	#v = np.mean(testdata[i])
	#guess = np.abs(app - v).argmin()
	guess = getMinDistanceIndex(app, testdata[i])
	if (guess != testlabel[i]):
		nbWrong[testlabel[i]]+=1
	#print("guess/actual : "+str(guess)+"/"+str(testlabel[i]))
	print(progressBar(i,len(testdata),60), end="\r")
	total[testlabel[i]]+=1

print(" "*60, end="\r")# erase progress line

bigWrong=0
bigTotal=0
for i in range(10):
	print("cat. "+str(i)+" taux d'erreur : "+str(round((nbWrong[i]/total[i])*100,2))+"%")
	bigWrong+=nbWrong[i]
	bigTotal+=total[i]
print("-----------------------------")
print("total  taux d'erreur : "+str(round((bigWrong/bigTotal)*100,2))+"%")


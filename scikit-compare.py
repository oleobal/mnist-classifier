from utility import *
import numpy as np
from sklearn import svm, metrics, neighbors
from scipy import sparse

displayWidth=39

print("Reading & training..")

data_unproc = np.load("data/trn_img.npy")
label = np.load("data/trn_lbl.npy")

totalLearningOps = 0
progress = 0

totalLearningOps+=28*28*10
#data = sparse.csr_matrix(data_unproc)
data = data_unproc

testdata_unproc = np.load("data/tst_img.npy")
testlabel = np.load("data/tst_lbl.npy")
#test = sparse.csr_matrix(testdata_unproc)

class classifier_struct:
    def __init__(self, classifier, name):
        self.classifier = classifier
        self.name = name

classifier_svc = classifier_struct(svm.SVC(kernel='linear'), "SVC")
classifier_neighbors = classifier_struct(neighbors.KNeighborsClassifier(), "KNeighbors")

classifiers = [classifier_svc, classifier_neighbors]

for classifier in classifiers:
    print("Training the classifier " + classifier.name)
    classifier.classifier.fit(data, label)
    print("Training done ")
    test = testdata_unproc
    print("Predicting...")
    expected = testlabel
    predicted = classifier.classifier.predict(test)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier.classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
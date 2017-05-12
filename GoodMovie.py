import sys
from sklearn import tree

#X = [[0, 0], [1, 1]] These are 2 inputs that include all of the features.
#Each input is a separate list of the values of the features
#Y = [0, 1] This is a column vector of the labels associated with each datapoint
#clf = tree.DecisionTreeClassifier() These 2 lines create the tree
#clf = clf.fit(X, Y)

trainingSet = []  #Contains the features of the training data
testingSet = []   #Contains the features of the testing set
trainingLabels = []   #Contains the labels of the training set
testingLabels = []    #Contains the predicted labels of the testing set

TP = 0  #True Positives
TN = 0  #True Negatives
FP = 0  #False Positives
FN = 0  #False Negatives
P = 0   #Positives
N = 0   #Negatives

trainingFile = sys.argv[1]  
testingFile = sys.argv[2]

with open(trainingFile) as f:
    firstLine = f.readline()  #Don't need first line
    for line in f.readlines():
        features = [int(num) for num in line.split()]
        trainingLabels.append(int(features[-1]))
        #Get rid of counter and label
        del features[0]
        del features[-1]
        trainingSet.append(features)

#print(trainingLabels)
#print(trainingSet)

clf = tree.DecisionTreeClassifier(random_state = 0)
clf = clf.fit(trainingSet, trainingLabels)                

with open(testingFile) as f:
    firstLine = f.readline() #Don't need first line
    for line in f.readlines():
        features = [int(num) for num in line.split()]
        testingLabels.append(int(features[-1]))
        #Get rid of counter and label
        del features[0]
        del features[-1]
        testingSet.append(features)

predicted = clf.predict(testingSet)

for i in range(len(testingLabels)):
    if(testingLabels[i] == 0):
        N += 1
        if(predicted[i] == 0):
            TN += 1
        else:
            FP += 1
            
    else:  #Else its a 1
        P += 1
        if(predicted[i] == 1):
            TP += 1
        else:
            FN += 1

errorRate = (FP + FN)/(P + N)
            
print("True positives = ", (TP))
print("True negatives = ", (TN))
print("False positives = ", (FP))
print("False negatives = ", (FN))
print("Error rate = ", (errorRate))



    

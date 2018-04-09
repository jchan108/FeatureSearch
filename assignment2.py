import argparse
import numpy as np
from scipy import stats
from numpy import linalg 
from operator import truediv
import itertools
import time

#KNN solver
def NearestNeighbor(data,example):
    
    traindata = np.vstack([data, example])
    xtraindata = traindata[:,1:] #just the features
    ztraindata = stats.zscore(xtraindata) #just the features 
    
    zdata = ztraindata[:ztraindata.shape[0]-1,:] #our training set
    zexample = ztraindata[ztraindata.shape[0]-1,:] #our test observation
    
    mindist = np.linalg.norm(zexample-zdata[0]) #initialize our mindist
    minindex = 0 #initialize our index of min distance

    for i in range(1,zdata.shape[0]): # 1 to 99
        curr = zdata[i]
        dist = np.linalg.norm(zexample-curr)
        if (dist < mindist):
            mindist = dist
            minindex = i

    if (example[0] == data[minindex,0]):
        return 1 #returns that NN predicted y correctly
    else:
        return 0 #retuns that NN predicted y wrong.
    
def leaveoneoutcv(datas,currfeat):
    
    subdata = datas[:,currfeat.astype('int64')] #every row only relevant columns
    correct = 0 #number of times knn gives correct answer

    for i in range(0,subdata.shape[0]): #for every single row (observation)  
        testset = subdata[i]
        trainset = np.delete(subdata,(i),axis=0)
        yhat = NearestNeighbor(trainset,testset) #test the example on the trainset
        correct = correct + yhat #adds either 1 or 0 to correct.    
        
    percent = correct/datas.shape[0] #number of right over number of rows
    return percent

def feature_search(data):
    currfeatures = np.array([0]) #0 is the y variable
    overallbest = 0
    
    print('Beginning Search.')
    for i in range(1,data.shape[1]):
        print('On level', i, 'of the search tree')
        #new_level_feature = 0 #new feature we are going to add to level to test
        bestacc = 0 #best accuracy on this level
        
        for j in range(1,data.shape[1]):
            #need to make sure no duplicates are searched
            
            if (j not in currfeatures ):
                testfeatures = np.append(currfeatures,j)#combine the list of features together
                accuracy = leaveoneoutcv(data,testfeatures)
                print('       Using feature(s)', testfeatures[1:], 'The accuracy is', accuracy)

                if (accuracy > bestacc):
                    bestacc = accuracy
                    new_level_feature = j
        
        currfeatures = np.append(currfeatures,new_level_feature)
        print ('Feature set',currfeatures[1:],'was the best, with an accuracy of',bestacc)
        if (bestacc > overallbest):
            overallbest = bestacc
            bestfeatureset = currfeatures
    print('Search is finished. The best feature set is',bestfeatureset[1:],'with an accuracy of',overallbest)

def leaveoneoutcv_prune(datas,currfeat,maxwrong):
    
    subdata = datas[:,currfeat.astype('int64')] #every row only relevant columns
    correct = 0 #number of times knn gives correct answer
    wrong = 0
    
    for i in range(0,subdata.shape[0]): #for every single row (observation)  
        testset = subdata[i]
        trainset = np.delete(subdata,(i),axis=0)
        yhat = NearestNeighbor(trainset,testset) #test the example on the trainset
        correct = correct + yhat #adds either 1 or 0 to correct.    
        if (yhat == 0):
            wrong = wrong + 1
        if (wrong > maxwrong): #prune
            return "pruned"
        
    return correct

def feature_search_prune(data):
    currfeatures = np.array([0]) #0 is the y variable
    overallbest = 0
    wrong = 100
    bestright = 0
    
    print('Beginning Search.')
    for i in range(1,data.shape[1]):
        print('On level', i, 'of the search tree')
        #new_level_feature = 0 #new feature we are going to add to level to test
        bestacc = 0 #best accuracy on this level
        
        for j in range(1,data.shape[1]):
            #need to make sure no duplicates are searched
            
            if (j not in currfeatures ):
                testfeatures = np.append(currfeatures,j)#combine the list of features together
                correct = leaveoneoutcv_prune(data,testfeatures,wrong)
                
                if (correct == "pruned"):
                    print('       Using feature(s)', testfeatures[1:], 'The accuracy is', correct)
                else:
                    accuracy = correct/data.shape[0]
                    print('       Using feature(s)', testfeatures[1:], 'The accuracy is', accuracy)
                    
                    if (accuracy > bestacc):
                            bestright = correct
                            bestacc = accuracy
                            new_level_feature = j
        
        currfeatures = np.append(currfeatures,new_level_feature)
        if (bestacc > overallbest):
            wrong = 100-bestright
            overallbest = bestacc
            bestfeatureset = currfeatures
            
            
        if (bestacc == 0):
            print("Terminating the forwards search early. Every feature set on this level was pruned")
            print('Search is finished. The best feature set is',bestfeatureset[1:],'with an accuracy of',overallbest)
            return
        
        
        print ('Feature set',currfeatures[1:],'was the best, with an accuracy of',bestacc)
    print('Search is finished. The best feature set is',bestfeatureset[1:],'with an accuracy of',overallbest)


def backwards_search(data):
    currfeatures = np.array(range(0,data.shape[1])) #every column including the y
    overallbest = 0
    print('Begining Backwards Elimination')
    
    entireaccuracy = leaveoneoutcv(data,currfeatures)
    print('Using the entire feature set', currfeatures[1:], 'we obtain an accuracy of',entireaccuracy)
    overallbest = entireaccuracy
    
    for i in range(1,data.shape[1]):
        print('On level', i, 'of the search tree')
        bestacc = 0 #the current best accuracy
        
        #feature_to_remove = 0 #the feature we remove at this level
        #print(currfeatures.shape)
        for j in range(1,currfeatures.shape[0]): #going over every index of the currfeatures array
            testfeatures = np.delete(currfeatures,j,axis = 0)#this should get a subset of # of columns - 1
            accuracy = leaveoneoutcv(data,testfeatures) #finds the accuracy of knn on just these features.
            print('       Using feature(s)', testfeatures[1:], 'The accuracy is', accuracy)
           # feature_to_remove = j
            
            if (accuracy >= bestacc):
                bestacc = accuracy
                bestset = testfeatures
                feature_to_remove = j
        print('Feature set', bestset[1:], 'was the best, with an accuracy of', bestacc)
        
        if (bestacc >= overallbest):
            overallbest = bestacc
            bestfeatureset = bestset
        currfeatures = np.delete(currfeatures,feature_to_remove,axis = 0)

    print('Search is finished. The best feature set is',bestfeatureset[1:], 'with an accuracy of', overallbest)

print('Welcome to my Feature Selection Album')
file = input('Type the name of the file you wish to test: \n')
data = np.loadtxt(file)

algorithm = input('Type the name of the algorithm you wish to run: \n 1) Forward Selection \n 2) Backward Elimination \n 3) My Special Algorithm \n')

start=time.clock()


if (algorithm == '1'):
    feature_search(data)
    end=time.clock()
    totaltime = end-start
    print("Time to conduct Forwards Search:", totaltime)

elif (algorithm == '2'):
    backwards_search(data)
    end=time.clock()
    totaltime = end-start
    print("Time to conduct Backwards Selection:", totaltime)
elif (algorithm == '3'):
    feature_search_prune(data)
    end=time.clock()
    totaltime = end-start
    print("Time to conduct Forwards Selection w/ Pruning:", totaltime)
else:
    tempy = 0
    end=time.clock()
    totaltime = end-start


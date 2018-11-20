from math import log
import operator
def calcShannoEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannoEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannoEnt-=prob*log(prob,2)
    return shannoEnt


def splitDataSet(dataset,axis,value):
    retDataSet=[]
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataset):
    numfeature=len(dataset[0])-1
    baseentropy=calcShannoEnt(dataset)
    bestinfogain=0.0
    bestfeature=-1
    for i in range(numfeature):
        featurelist=[example[i] for example in dataset]
        uniquevals=set(featurelist)
        newentropy=0.0
        for value in uniquevals:
            subdataset=splitDataSet(dataset,i,value)
            prob=len(subdataset)/float(len(dataset))
            newentropy+=prob*calcShannoEnt(subdataset)
        infogain=baseentropy-newentropy
        if (infogain>bestinfogain):
            bestinfogain=infogain
            bestfeature=i
    return bestfeature


def majorityCnt(classlist):
    classcount={}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote]=0
        classcount[vote]+=1
    sortedclasscount=sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

def creattree(dataset,label):
    classlist=[example[-1] for example in dataset]
    if classlist.count(classlist[0])==len(classlist):
        return  classlist[0]
    if (len(dataset[0])==1):
        return  majorityCnt(classlist)
    bestfeat=chooseBestFeatureToSplit(dataset)
    bestfeatlabel=label[bestfeat]
    mytree={bestfeatlabel:{}}
    del (label[bestfeat])
    featvalue=[example[bestfeat] for example in dataset]
    uniquevals=set(featvalue)
    for value in uniquevals:
        sublabel=label[:]
        mytree[bestfeatlabel][value]=creattree(splitDataSet(dataset,bestfeat,value),sublabel)
    return mytree

def classify(inputtree,featlabels,testvec):
    keylist=list(inputtree.keys())
    firststr=keylist[0]
    secondDict=inputtree[firststr]
    featindex=featlabels.index(firststr)
    for key in secondDict.keys():
        if testvec[featindex]==key:
            if type(secondDict[key]).__name__=='dict':
                classlabel=classify(secondDict[key],featlabels,testvec)
            else:
                classlabel=secondDict[key]
    return classlabel



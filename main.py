# -*- coding: UTF-8 -*-
import tree
import creatDataSet

myDat,labels=creatDataSet.creatDataSet()
print (myDat)

#myDat[0][-1]='maybe'
print (tree.calcShannoEnt(myDat))
#tree.chooseBestFeatureToSplit(myDat)
best=tree.chooseBestFeatureToSplit(myDat)
print (labels[best])
mytree=tree.creattree(myDat,labels)
#print (mytree)
print (tree.classify(mytree,labels,[1,1,2,2,2,1]))


#Robert Genega 
#cs471 project 3

#machine_learning


import sys
from _io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


class node(object):
    def __init__(self):
        self.name=None
        self.node=[]
        self.columnNum = None
        self.columnData = None
        self.prev=None
        self.label = None
        self.leaf = 0
    def next(self,child):
        #"Gets a node by number"
        return self.node[child]
    def prev(self):
        return self.prev
    def goto(self,data):
        #s"Gets the node by name"
        for child in range(0,len(self.node)):
            if(self.node[child].name==data):
                return self.node[child]
    def add(self):
        node1=node()
        self.node.append(node1)
        node1.prev=self
        return node1
    def setNodeData(self, colnum, coldata):
        self.columnNum = colnum
        self.columnData = coldata

#calculates entropy of two distinct elements (number of each element)
def entropy(a, b):
    tot = a + b

    if a == 0 or b == 0:
        return 0


    val = (a/tot)*math.log(a/tot, 2) + (b/tot)*math.log(b/tot, 2)
    return -val

#calculates entropy of two columns
def entropyTable(a, b):
    unique = []
    for element in a:
        if element not in unique:
            unique.append(element)

    ucount = []
    fcount = []
    for element in unique:
        ucount.append( a.count(element) )
        fcount.append(0)

    tot = sum( ucount )
    uprob = []

    for element in ucount:
        uprob.append(element/tot)


    i = 0
    for element in a:
        fcount[unique.index(element)] += b[i]
        i+=1

    uentropy = []

    i= 0
    for element in fcount:

        uentropy.append( entropy(element, ucount[i]-element) )
        i+=1

    i = 0
    finale = 0
    for element in uprob:
        finale += element * uentropy[i]
        i+=1



    return finale

#prints decision tree created for debugging purposes
def printTree(node):
    parentName = 99

    if node.prev != None:
        parentName = node.prev.name
    print(parentName, node.name, "COL: ", node.columnNum)
    for element in node.node:
        print("Child Name = ", element.name)

    for element in node.node:
        printTree(element)

#builds the decision tree given the training table and the max depth of the decision tree
def buildTree(table, node, depth=1):
    #print(table)

    entropyList = []
    
    rowList = list(table.index)
    columnList = list(table.columns.values)

    lastColumnIndex = columnList[-1]
    
    lastColumnData = table[lastColumnIndex].tolist()

    columnList.pop()
    for i in columnList:
        q = table[i]
        a = q.tolist()

        #a = table[i].tolist()

        entropyList.append( entropyTable( a, lastColumnData ) )

    minEntropyColumn = entropyList.index(min(entropyList))
    col = table[ columnList[minEntropyColumn] ].tolist()

    unique = []
    for element in table[ columnList[minEntropyColumn] ].tolist():
        if element not in unique:
            unique.append(element)
            

    node.setNodeData(columnList[minEntropyColumn], unique)

    for element in unique:
        child = node.add()
        child.name = element

        #entropy is 0 case
        if entropyList[minEntropyColumn] == 0:
            i = col.index(element)
            child.label = lastColumnData[i]
            child.leaf = 1
            

        #no columns left case
        elif len(table.columns) == 2 or depth == 1:
            labelcount = 0
            uniquetotal = 0

            for i in range(len(lastColumnData)):
                if col[i] == element:
                    uniquetotal += 1
                    if lastColumnData[i] == 1:
                        labelcount += 1

            child.label = round(labelcount/uniquetotal)
            child.leaf = 1
            

        #columns left case and entropy is not 0
        else:

            labelcount = 0
            uniquetotal = 0

            for i in range(len(lastColumnData)):
                if col[i] == element:
                    uniquetotal += 1
                    if lastColumnData[i] == 1:
                        labelcount += 1

            child.label = round(labelcount/uniquetotal)
            child.leaf = 0


            #cToDrop = table.columns[[columnList[minEntropyColumn]]]
            cToDrop = table.columns[minEntropyColumn]
            table1 = table.drop(cToDrop, axis=1)
            #print(table1)

            rowstodrop = []

            for i in range(len(lastColumnData)):
                if col[i] != element:
                    rowstodrop.append(rowList[i])
            
            #tt1 = table1.index[rowstodrop]
            dict = {}
            for temp in table.index:
                if type(temp) == 'str':
                    aa = temp.strip()
                else:
                    aa = temp
                dict[temp] = aa

            for index in rowstodrop:
                indexToDrop = dict[index]
                if indexToDrop != None:
                    table1 = table1.drop(indexToDrop)

            #table1 = table1.drop(tt1)

            buildTree(table1, child, depth-1)



            

    return



#beginning of MAIN

train_file = sys.argv[1]
input_depth = int(sys.argv[2])


#converts train_file input to pandas dataframe
#df = pd.read_csv(train_file, header=None, index_col = False)






#clean titanic train file
cleanedFile = ""
testFile = ""

file = open(train_file, 'r')
train_vector = []

isfirst = 0

lineCount = 0

for line in file:
    tokens = line.split(",")

    if isfirst != 0:

        currCol = 0
        for element in tokens:
            
            if currCol == 0 or currCol == 3 or currCol == 4 or currCol == 9 or currCol == 10 or currCol == 11:
                pass
            elif currCol == 1 or currCol == 2 or currCol == 7 or currCol == 8:
                train_vector.append( int(element) )
            elif currCol == 5:
                if element == "male":
                    train_vector.append(0)
                else:
                    train_vector.append(1)
            elif currCol == 6:
                if element == "":
                    train_vector.append(4)
                else:
                    age = int( float(element) )

                    if age < 7:
                        age = 0
                    elif age < 14:
                        age = 1
                    elif age < 21:
                        age = 2
                    elif age < 28:
                        age = 3
                    elif age < 35:
                        age = 4
                    else:
                        age = 5

                    train_vector.append( age)
            elif currCol == 12:
                if element == "S\n":
                    
                    train_vector.append(1)
                elif element == "C\n":
                    
                    train_vector.append(2)
                elif element == "Q\n":
                    
                    train_vector.append(3)
                else:
                    #set port to S (most common)
                    train_vector.append(1)

            else:
                train_vector.append(element)

            currCol += 1
        train_vector.append( train_vector.pop(0) )

    #print(train_vector)

    #number of entries removed from training set and used 
    #for the test set
    testSize = 20

    for element in train_vector:
        if lineCount <= 891 - testSize:
            cleanedFile += str(element)
            cleanedFile += ","
        else:
            testFile += str(element)
            testFile += ","

    if lineCount <= 891 - testSize:
        cleanedFile = cleanedFile[:-1]
        cleanedFile += "\n"
    else:
        testFile = testFile[:-1]
        testFile += "\n"

    train_vector = []
    isfirst += 1

    lineCount+=1

#print(cleanedFile)

#print(testFile)



    

file.close()




file1 = open("temp.txt", "w")

file1.write(cleanedFile)

file1.close()

df = pd.read_csv("temp.txt", header=None, index_col = False)

#print(df)



tree=node()

buildTree(df, tree, input_depth)
#printTree(tree)





#READ in input test vector
file = open("foo.txt", 'w')

file.write(testFile)
file.close()

file = open("foo.txt", 'r')

input_vector = []



for line in file:

    tokens = line.split(",")
    
    reader = []
    
    for element in tokens:
        
        reader.append(int(element))
    input_vector.append(reader)
        

file.close()

#print(input_vector)




#navigate decision tree and print guess


for element in input_vector:
    currNode = tree
    prevNode = None
    while currNode != None and currNode.leaf == 0:
        currCol = currNode.columnNum

        name = element[currCol]
        prevNode = currNode
        currNode = currNode.goto(name)


    if currNode != None:
        print(currNode.label)
    else:
        print(prevNode.label)

















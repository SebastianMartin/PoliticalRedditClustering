#! usr/bin/env python3
import sys, os
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt


#look at this and figure out how it works
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import numpy as np
from scipy.spatial import ConvexHull

import numpy
from scipy import sparse
def getListOfEachSubreddit():
	dataList = []
	dataPath = "../data"
	dirrec1 = os.listdir(dataPath)


	for files in dirrec1:
		allPosts = []
		#print(dataPath+"/"+files)
		dataFile = dataPath+"/"+files
		currentFile = []
		openFile = open(dataFile,"r")
		lines = ""
		for line in openFile:
			currentPost = []
			splitLine1 = line.split("<'|'>")
			textLine = splitLine1[1]+" "+splitLine1[7]
			line2 = textLine.translate(str.maketrans('','', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,,.,<,>,[,],/,?,\\,|,`,~,\n"))
			splitLine = line2.split(" ")
			if splitLine != ['\n']:
				for word in splitLine:
					currentPost.append(word.lower())
			allPosts.append(currentPost)
		openFile.close()
		dataList.append(allPosts)
	print(len(dataList))
	for l in dataList:
		print(len(l))
	return dataList,dirrec1
def cleanList(line):
	for i in range(len(line)-1):
		#print(line[i])
		if line[i] == '':
			line.pop(i)
			cleanList(line)
			break


	return line
def makeListIntoPairs(listOfStrings,Name):
	for line in listOfStrings:
		line = cleanList(line)


	allPairs = []
	commonPairs = [('of', 'the'),('in', 'the'),('is', 'a'),('to', 'the'),
	('is', 'the'),('on', 'the'),('to', 'be'),
	('it', ''),('this', 'is'),('to', 'be'),('in', 'a'),('and', 'the'),
	('is', 'not'),('for', 'the'),('it', 'is'),('the', 'us'),('with', 'a'),('as', 'a'),
	('on', 'this'),('if', 'you'),('to', 'a'),('gun', ''),('guns', ''),
	('to', 'get'),('here', ''),('true', ''),('we', 'are'),('with', 'the'),
	('that', 'the'),('from', 'the') ,('at', 'the') ,('by', 'the') 
	]
	for line in listOfStrings:

		for i in range(len(line)-1):
			try:
				if (line[i],line[i+1]) not in commonPairs:
					allPairs.append((line[i],line[i+1]))
			except:
				print("wut?")
	freqList = {i:allPairs.count(i) for i in set(allPairs)}
	sortedDict = sorted(freqList.items(), key=lambda item: item[1],reverse = True)
	print("2 Pair")
	for i in range(10):
		print(sortedDict[i][0],"\t",sortedDict[i][1])
def makeListIntoTriplets(listOfStrings,Name):
	for line in listOfStrings:
		line = cleanList(line)


	allPairs = []
	commonPairs = [('of', 'the'),]
	for line in listOfStrings:

		for i in range(len(line)-2):
			try:
				if (line[i],line[i+1],line[i+2]) not in commonPairs:
					allPairs.append((line[i],line[i+1],line[i+2]))
			except:
				print("wut?")
	freqList = {i:allPairs.count(i) for i in set(allPairs)}
	sortedDict = sorted(freqList.items(), key=lambda item: item[1],reverse = True)
	print("3 Pair")
	for i in range(10):
		print(sortedDict[i][0],"\t",sortedDict[i][1])
def makeListInto4lets(listOfStrings,Name):
	for line in listOfStrings:
		line = cleanList(line)


	allPairs = []
	commonPairs = [('of', 'the'),]
	for line in listOfStrings:

		for i in range(len(line)-3):
			try:
				if (line[i],line[i+1],line[i+2],line[i+3]) not in commonPairs:
					allPairs.append((line[i],line[i+1],line[i+2],line[i+3]))
			except:
				print("wut?")
	freqList = {i:allPairs.count(i) for i in set(allPairs)}
	sortedDict = sorted(freqList.items(), key=lambda item: item[1],reverse = True)
	print("4 Pair")
	for i in range(10):
		print(sortedDict[i][0],"\t",sortedDict[i][1])

def main():
	allRedditLists,NameList = getListOfEachSubreddit()
	for i in range(len(allRedditLists)):
		print(NameList[i])
		makeListIntoPairs(allRedditLists[i],NameList[i])
		makeListIntoTriplets(allRedditLists[i],NameList[i])
		makeListInto4lets(allRedditLists[i],NameList[i])



main()

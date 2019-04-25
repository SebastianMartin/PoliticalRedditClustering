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

fullListFreq = {}


def removeStops(essay,stopWords):
	newEssay = []
	for word in essay:
		if word not in stopWords and word != "":
			newEssay.append(word)
	return newEssay

def createStopList():
	stopWords = []
	f0 = open("stop.txt","r")
	for line in f0:
		word = line.replace("\n","")
		stopWords.append(word)
	return stopWords
	f0.close()
def createDictionaryForFile(file):
	freq = {}
	for line in file:
		for word in line.strip().split():
			if word not in freq:
				freq[word] = 0
			freq[word] += 1
			if word not in fullListFreq:
				fullListFreq[word] = 0
			fullListFreq[word] += 1
			'''if word not in fullWordList:
				fullWordList.append(word)'''
	sortedDict = sorted(freq.items(), key=lambda item: item[1],reverse = True)
	return freq
def getAllPosts():
	dataList = []
	dataPath = "../data"
	dirrec1 = os.listdir(dataPath)
	allPosts = []
	'''for i in range(1):
		if dirrec1[i] != ".DS_Store":
			path = "Data2014/"+dirrec1[i]'''
	print(dirrec1)

	for files in dirrec1:
		#print(dataPath+"/"+files)
		dataFile = dataPath+"/"+files
		currentFile = []
		openFile = open(dataFile,"r")
		lines = ""
		for line in openFile:
			currentPost = []
			splitLine1 = line.split("<'|'>")
			textLine = splitLine1[1]+" "+splitLine1[7]
			line2 = textLine.translate(str.maketrans('','', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,,.,<,>,/,?,\\,|,`,~,\n"))
			splitLine = line2.split(" ")
			if splitLine != ['\n']:
				for word in splitLine:
					currentPost.append(word.lower())
			allPosts.append(currentPost)
		openFile.close()
	return allPosts,dirrec1

def tfComputation(docFreqList, document):
	tf = {}
	length = len(document)
	for word, count in docFreqList.items():
		tf[word] = count/length
	return tf
def computeTDF(essayList):
	n = len(essayList)
	idf = {}
	for l in essayList:
		for word, count in l.items():
			if count > 0:
				if word not in idf:
					idf[word] = 0
				idf[word] += 1
	
	for word, v in idf.items():
		idf[word] = math.log(n / float(v))
	return idf
def computeTFIDF(tf,idf):
    tf_idf = dict.fromkeys(idf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return list(map(list, tf_idf.items()))


def getCombinedTFIDF(cluster,allTFIDFS):
	newDict = dict(fullListFreq)
	for word in newDict:
		newDict[word] = 0
	dictAsList = list(map(list, newDict.items()))
	for x in range(len(cluster)):
		for i in range(len(dictAsList)):
			dictAsList[i][1] += allTFIDFS[int(cluster[x])][i][1]
	dictAsList.sort(key = lambda x: x[1], reverse = True) 
	return dictAsList


def main():
	start = time.time()
	dotWord = "...................................................."
#-----------------------------------------------------------------------------------------------------------
	#get List of all stop words
	statement = "create stop word list"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)

	stopWords = createStopList()
	print(stopWords)
	end1 = time.time()

	print("DONE\n")
	print("Time to load stop words  (seconds):\t",round(end1 - start,15))
	
#----------------------------------------------------------------------------------------------------------
	#get 2d list of all words in all posts
	statement = "Loading all Posts"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	allPosts,dataList = getAllPosts()
	print("DONE\n")
	end1 = time.time()
	print("Time to load all Posts   (seconds):\t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	#remove all of the stop words within each essay
	statement = "Remove all stop words from Posts"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	for i in range(len(allPosts)):
		allPosts[i] = removeStops(allPosts[i],stopWords)
	print("DONE\n")
	end1 = time.time()
	print("Time to remove stop words(seconds):\t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	#convert each essay into a list of (word,freq) tuples
	statement = "Convert each post to freqCounts"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	freqCounts = []
	for post in allPosts:
		freqCounts.append(createDictionaryForFile(post))
	print("DONE\n")
	end1 = time.time()
	print("Time to create post count(seconds):\t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	#Get the IDF of all the terms
	statement = "Get the IDF computation"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	idf = computeTDF(freqCounts)
	print("DONE\n")
	end1 = time.time()
	print("Time to create IDF       (seconds):\t",round(end1 - start,15))

#-----------------------------------------------------------------------------------------------------------
	#Get the IDF of all the terms
	statement = "Get the TF computations"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	tfComputations = []
	for i in range(len(allPosts)):
		tfComputations.append(tfComputation(freqCounts[i],allPosts[i]))
	print("DONE\n")
	end1 = time.time()
	print("Time to create TF's      (seconds):\t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	#Get the TF-IDF of each document
	statement = "Get the TF-IDF computations"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	tfIDFs = []
	for tf in tfComputations:
		tfIDFs.append(computeTFIDF(tf,idf))
	print("DONE\n")
	end1 = time.time()
	print("Time to create  TF-IDF's (seconds):\t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	#Get the full matrix
	statement = "Generate complete matrix's"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	matrix = []
	for post in tfIDFs:
		docLine = []
		for word in post:
			docLine.append(word[1])
		matrix.append(docLine)
	docMatrixDense = numpy.matrixlib.defmatrix.matrix(matrix)
	docMatrixSparse = sparse.csr_matrix(docMatrixDense)
	print("DONE\n")
	end1 = time.time()
	print("Time to create Matrix    (seconds):\t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	#reduce the dimensions of the
	statement = "Reduce the dimensions to 2"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	matrix = []

	#reduced_data = PCA(n_components=2).fit_transform(docMatrixDense)
	reduced_data = TSNE(n_components=2).fit_transform(docMatrixDense)

	print("DONE\n")
	end1 = time.time()
	print("Time to reduce dimensions(seconds):\t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------

	#doing kmeans
	statement = "doing kMeans"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)

	clusterSize = 9
	kmeans_model = KMeans(n_clusters=clusterSize).fit(reduced_data)
	labels = kmeans_model.labels_
	centers = np.array(kmeans_model.cluster_centers_)

	print("DONE\n")
	end1 = time.time()
	print("Time to create kMeans    (seconds):\t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------

	#put all of the documents that were connected together
	statement = "grouping Clusers"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)

	clusterLists = [[] for _ in range(clusterSize)]
	for i in range(len(labels)):
		clusterLists[labels[i]].append(str(i))

	print("DONE\n")
	end1 = time.time()
	print("Time to group clusters  (seconds):\t",round(end1 - start,15))

#-----------------------------------------------------------------------------------------------------------

	#go through each cluster and combine all of the tfidfs that we found
	statement = "grouping TF-IDF's"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)

	clusterTFIDFS = []
	for cluster in clusterLists:
		clusterTFIDFS.append(getCombinedTFIDF(cluster,tfIDFs))

	print("DONE\n")
	end1 = time.time()
	print("Time to group TF-IDF's  (seconds):\t",round(end1 - start,15))



#-----------------------------------------------------------------------------------------------------------

	#Printing the plots and the chart and the hull

	# print reduced_data
	labels_color_map = {
	    0: '#c687c5', 1: '#ff0000', 2: '#ffbb00', 3: '#94ff00', 4: '#7c8ea5',
	    5: '#0090ff', 6: '#7200ff', 7: '#f600ff', 8: '#44440d', 9: '#7c8ea5'
	}
	fig, ax = plt.subplots()
	postCount = 500
	for index, instance in enumerate(reduced_data):
		found = False
		k = 0
		while found == False:
			for i in range(postCount):
				#print(i)
				if (index) == (postCount*k+i):
					found = True
			k+=1
		#print(k)

		# print instance, index, labels[index]
		pca_comp_1, pca_comp_2 = reduced_data[index]
		color = labels_color_map[k-1]
		if( (index%postCount) == 0):
			#print(index/postCount)
			ax.scatter(pca_comp_1, pca_comp_2 , c=color, s=15,label = "r/"+dataList[int(index/postCount)][:-4])

		else:
			ax.scatter(pca_comp_1, pca_comp_2 , c=color, s=15)
		#ax.annotate(index,(pca_comp_1, pca_comp_2))

	plt.scatter(centers[:,0], centers[:,1], marker="x", color='r',s = 100)




	clusterPoints = [[] for _ in range(clusterSize)] 
	for i in range(len(labels)):
		clusterPoints[labels[i]].append(reduced_data[i])
	for i in range(len(clusterPoints)):
		clusterArray = np.asarray(clusterPoints[i])		
		hull = ConvexHull(clusterArray)
		for simplex in hull.simplices:  
			plt.plot(clusterArray[simplex, 0], clusterArray[simplex, 1], labels_color_map[i])#"#000000")
		plt.plot(clusterArray[hull.vertices,0], clusterArray[hull.vertices,1], labels_color_map[i])#"#000000")#"cluster: " + str(i) + "  |  Area: "+ str(hull.area))
	plt.gca().legend()



	emptyWord = "                                      "
	print(len(clusterTFIDFS))
	dashWord = "________________________________"
	for j in range(len(clusterTFIDFS)-1):
		currWord = "cluster-"+str(j)
		print(currWord+dashWord[:22-len(currWord)] + "|", end ="",flush=True )
	print("cluster-"+str(len(clusterTFIDFS)-1)+dashWord[:22-len("cluster-"+str(len(clusterTFIDFS)-1))])
	for i in range(15):
		for j in range(len(clusterTFIDFS)-1):
			word = clusterTFIDFS[j][i][0]
			print(word+emptyWord[:22-len(word)] + "|", end ="",flush=True )
		print(clusterTFIDFS[len(clusterTFIDFS)-1][i][0])
	allClusterTopSubs = []
	for cluster in clusterLists:
		#print(cluster)
		currentCluster = []
		for data in dataList:
			currentCluster.append([data,0])
		#print(currentCluster)
		for point in cluster:
			subReddit = int(float(point)/postCount)
			#print(point,"\t",subReddit)
			#count = currentCluster[subReddit][1]+1
			currentCluster[subReddit][1]+=1# = count
		#print(currentCluster)
		currentCluster.sort(key=lambda x: x[1],reverse=True)
		allClusterTopSubs.append(currentCluster)

		#print(cluster)
	print("\n\n")
	emptyWord = "                                      "
	dashWord = "________________________________"
	for j in range(len(clusterTFIDFS)-1):
		currWord = "cluster-"+str(j)
		print(currWord+dashWord[:22-len(currWord)] + "|", end ="",flush=True )
	print("cluster-"+str(len(clusterTFIDFS)-1)+dashWord[:22-len("cluster-"+str(len(clusterTFIDFS)-1))])
	for i in range(9):
		for j in range(8):
			word = allClusterTopSubs[j][i][0][:-4]+" "+str(allClusterTopSubs[j][i][1])
			#print(word)
			print(word+emptyWord[:22-len(word)] + "|", end ="",flush=True )
		print(allClusterTopSubs[len(allClusterTopSubs)-1][i][0][:-4]+" "+str(allClusterTopSubs[len(allClusterTopSubs)-1][i][1]))




	end1 = time.time()

	print("Time to complete        (seconds):\t",round(end1 - start,15))
	plt.show()














main()

#! /usr/bin/python3
from collections import defaultdict
import sys
import time
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import importlib

import topologies_2

global switchPathCount, pdcPathCount
switchPathCount = dict()
linkPathCount = dict()
pdcPathCount = dict()

def DFS_SP(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in set(graph[vertex]) - set(path):
            if next == goal:
                return path + [next]
                #yield path
            else:
                stack.append((next, path + [next]))



def BFS_SP(graph, start, goal):
	explored = []
	
	# Queue for traversing the
	# graph in the BFS
	queue = [[start]]
	
	# If the desired node is
	# reached
	if start == goal:
		print("Same Node")
		return
	
	# Loop to traverse the graph with the help of the queue
	while queue:
		path = queue.pop(0)
		node = path[-1]
		
		# Condition to check if the current node is not visite
		if node not in explored:
			neighbours = graph[node]
			
			# Loop to iterate over the neighbours of the node
			for neighbour in neighbours:
				new_path = list(path)
				new_path.append(neighbour)
				queue.append(new_path)
				
				# Condition to check if the neighbour node is the goal
				if neighbour == goal:
					return new_path 
			explored.append(node)

	return None

def getNonAffectedBusMapping(d, keys):
        return {x: d[x] for x in d if x not in keys}

def getObservableBusList(pmuBusMapping, affectedPMUs):
	observableBuses = set()
	unObservableBuses = set()
	
	# get a dictionary list of unaffected PMUs
	nonAffectedBusMapping = getNonAffectedBusMapping(pmuBusMapping, affectedPMUs)

	# Check if affected buses present in any of other unaffected PMUs
	for pmu in affectedPMUs:
		for bus in pmuBusMapping[pmu]:
			if any(bus in val for val in nonAffectedBusMapping.values()):
				observableBuses.add(bus)
			else:
				unObservableBuses.add(bus)

	return observableBuses, unObservableBuses

def getSwitchPathCount(path, switchList, pdcList):
	for switch in switchList:
		if switch in path:
			switchPathCount[switch][0] += 1
	for pdc in pdcList:
		if pdc in path:
			pdcPathCount[pdc] += 1

def getLinkPathCount(path, linksDown):
	linkSet = [set(val) for val in linksDown]
	for i in range( len(path) - 1):
		link = set([path[i], path[i+1]])
		if link in linkSet:
			linkPathCount[str(linksDown[linkSet.index(link)])] += 1
			
# Driver Code
if __name__ == "__main__":

	# Get required network from the topologies.py file 
    
	pmuNetwork, masterPathList, pmuBusMapping, numBuses = topologies_2.bus118()

	# Dictionary to store observability for each link down combination
	observabilityDict = dict()

	linksDown = []
	#numAltPaths = 0
	numAltPaths = 1
	Count=0

	t1 = time.time()
	for node in pmuNetwork:
		for i in range(len(pmuNetwork[node])):

			# To include only Switch-Switch links 
			#if set([node,pmuNetwork[node][i]]) not in linksDown and 'P' not in node and 'P' not in graph[node][i] :
                        if set([node,pmuNetwork[node][i]]) not in linksDown:

			# To include both Switch-Switch and PMU-Switch, but not PDC-Switch
			#if set([node,pmuNetwork[node][i]]) not in linksDown and 'D' not in node and 'D' not in graph[node][i]:
                            linksDown.append(set([node,pmuNetwork[node][i]]))


	# get list of switches and initilize number of paths and degree of each switch
	switchList = [val for val in pmuNetwork if 'S' in val]

	print("switchList = ", switchList)
	linksDown = [list(link) for link in linksDown]
	numLinks = len(linksDown)
	
	for switch in switchList:
		switchPathCount[switch] = []
		switchPathCount[switch].append(0) #initialize number of paths to 0
		switchPathCount[switch].append(len(pmuNetwork[switch])) #append degree to respective switch

	# Initialize Num of paths per pdc to 0
	pdcList = [val for val in pmuNetwork if 'PD' in val]
	for pdc in pdcList:
		pdcPathCount[pdc] = 0

	# Initialize Num of paths per link to 0
	for link in linksDown:
		linkPathCount[str(link)] = 0
	
	linksDownComb = []
	if sys.argv[1] == '2links':
		for i in range(0, numLinks - 1):
			for j in range(i+1, numLinks):
				linksDownComb.append([linksDown[i],linksDown[j]])
	
	else:
		singleLinks = [[val] for val in linksDown]
		linksDownComb.extend(singleLinks)
	
	print("---------------------------------------------------------")
	print("<Set of all links>\n")
	print("Total = ", numLinks,"\n")
	#print(*linksDown, sep="\n")
	print(linksDown)
	print("---------------------------------------------------------")

	print("---------------------------------------------------------")
	print("<Set of Link Combinations>\n")
	print("Total = ", len(linksDownComb),"\n")
	print(*linksDownComb, sep="\n")
	print("---------------------------------------------------------\n")
	
	if sys.argv[2] == 'bfs':
		print("\nBFS paths\n")
	if sys.argv[2] == 'dfs':
		print("\nDFS paths\n")
	for linkComb in linksDownComb:
		affectedPMUs = []
		graphTemp = copy.deepcopy(pmuNetwork)

		for link in linkComb:
			for pmu in masterPathList:
				#if link[0] in masterPathList[pmu] and link[1] in masterPathList[pmu] and pmu not in affectedPMUs:
				if ((link[0] in masterPathList[pmu] and link[1] in masterPathList[pmu]) or (link[0]==pmu and link[1] in masterPathList[pmu]) or (link[1]==pmu and link[0] in masterPathList[pmu])) and pmu not in affectedPMUs:
					affectedPMUs.append(pmu)

		print("Link Combination : ", linkComb, "\n\nPotentially affected PMUs = ",end='') 
		print(*affectedPMUs)

		for link in linkComb:
			graphTemp[link[0]].remove(link[1])
			graphTemp[link[1]].remove(link[0])

		for pmu in affectedPMUs[-1::-1]:
			print("\nPaths for",pmu,":")
			tempPdcList = copy.deepcopy(pdcList)

			while tempPdcList != []:
				pdc = random.choice(tempPdcList) # choose a random PDC as destination for checking path

				# Run BFS/DFS between each affected PMU and PDC
				if sys.argv[2] == 'dfs':
					path = DFS_SP(graphTemp, pmu, pdc)
				else:
					path = BFS_SP(graphTemp, pmu, pdc)
				Count += 1

				if path != None:
					print(*path,sep=' -> ')
					affectedPMUs.remove(pmu) #remove pmu from affected list, if path exists
					getSwitchPathCount(path, switchList, pdcList) # increment path count for each involved switch
					getLinkPathCount(path, linksDown) # increment path count for each involved link
					numAltPaths+=1
					break

				if path == None:
					tempPdcList.remove(pdc) #remove pdc from tempPdcList
			if tempPdcList == []:
				print("No alt paths!")

		print("\nFinal Affected PMUs = ", *affectedPMUs)
		observableBuses, unObservableBuses = getObservableBusList(pmuBusMapping, affectedPMUs)
		observabilityDict[str(linkComb)] = round(((numBuses - len(unObservableBuses)) * 100) / numBuses, 2)

		print("\nObservable buses : ", list(observableBuses))
		print("Un-Observable buses : ", list(unObservableBuses))
		print("Obs % :",observabilityDict[str(linkComb)])
		print("-----------------------------------------------\n")
	
	t2 = time.time()
	
	# Analysis 	
	print("Exec time : " + str("{:0.2f}".format(1000 * (t2-t1))) +  " ms")
	print("# times BFS/DFS called : {}".format(Count))

	print("\nObs % for each link down Combination:\n")
	d_view = [ (v,k) for k,v in observabilityDict.items()]
	d_view.sort()
	for v,k in d_view:
		print("%s:\t\t %10.2f" % (k,v))

	print("\n# of alternate paths, degree per switch") 
	print("-----------------------------------------")
	print("Switch\tDeg\tPaths\t% Util")
	print("----------------------------")
	s_view = [ (v[0],v[1],k) for k,v in switchPathCount.items()]
	s_view.sort()
	for v0,v1,k in s_view:
		print("%s:\t%d\t%d\t\t%.2f" % (k,v1,v0,100*v0/numAltPaths))


	print("\n# of alternate paths per PDC") 
	print("---------------------------------")
	print("PDC\tPaths")
	print("--------------")
	[print(key + ':',value,sep='\t\t') for key, value in pdcPathCount.items()]

	print("\n# of alternate paths per link") 
	print("--------------------------------------")
	print("Link\t\tPaths\t\t% Util")
	print("--------------------------------------")
	d_view = [ (v,k) for k,v in linkPathCount.items()]
	d_view.sort()
	for v,k in d_view:
		print("%s:\t%d\t\t%.2f" % (k,v,100*v/numAltPaths))

	# Plot Histogram and CDF
	obsList = [val for key, val in observabilityDict.items()]
	print("Obs-List", obsList)
	count, bins= np.histogram(obsList)
	pdf = count / sum(count)
	cdf = np.cumsum(pdf)
#	plt.figure(figsize=(20,5))

	plt.subplot(2, 2, 1)
	plt.plot(bins[1:], cdf, label="CDF")
	plt.legend()

	plt.subplot(2, 2, 2)

	plt.xlabel("Observability %")
	plt.ylabel("# Link Failure Combinations")
	plt.tight_layout()
	rangeStart = int(min(obsList)) - 10
	freq, bins, patches = plt.hist(obsList, edgecolor='white', color='g', bins=range(rangeStart,101,1))

	# x coordinate for labels
	bin_centers = np.diff(bins)*0.5 + bins[:-1]

	n = 0
	plt.annotate("Total Comb: %d" % (len(linksDownComb)), xy=(rangeStart,3))
	for fr, x, patch in zip(freq, bin_centers, patches):
		height = int(freq[n])
		if height!= 0:
			plt.annotate("%d (%0.2f %s)" % (height, 100 * height/len(linksDownComb), "%"), xy = (x, height),             # top left corner of the histogram bar
	       xytext = (0,0.2),             # offsetting label position above its bar
	       textcoords = "offset points", # Offset (in points) from the *xy* value
	       ha = 'center', va = 'bottom'
	       )
		n = n+1

	#plt.legend()

	plt.subplot(2, 2, 3)
	linkNames = [k for v,k in d_view]
	values = [v for v,k in d_view]
	plt.bar(range(len(d_view)), values, tick_label=linkNames, width=0.8, hatch='/')
	plt.xlabel("Link")
	plt.ylabel("No. of Alt. Paths")
	plt.xticks(rotation=45)
	plt.tight_layout()

	plt.subplot(2, 2, 4)
	switches = [k for v0,v1,k in s_view]
	values = [v0 for v0,v1,k in s_view]
	plt.bar(range(len(s_view)), values, tick_label=switches, width=0.8, hatch='/')
	plt.xlabel("Switch")
	plt.ylabel("No. of Alt. Paths")
#	plt.xticks(rotation=45)
	plt.tight_layout()

	plt.show()

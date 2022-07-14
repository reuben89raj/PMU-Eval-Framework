#! /usr/bin/python3
from collections import defaultdict
import sys
import time
import copy
import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams.update({'font.size': 10})
#plt.rc('axes', labelsize=12)

import numpy as np
import seaborn as sns

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
		
def plotObs(obsList, numBuses):
	N = numBuses
	list_count = [0] * N
	obsPercent = [0] * N

	for i in range(0, N):
		obsPercent[i]=round((100*(i+1)/N),2)
		list_count[i]=obsList.count(obsPercent[i])	

	print("list count = ", *list_count)	
	startIndex=list_count.index(not 0)

	list_plt = list_count[startIndex::]
	obsPercent_plt = obsPercent[startIndex::]
	obsPercent_plt = [round(val,1) for val in obsPercent_plt]
	M = N - startIndex
	ind = np.arange(M)

	width = 0.3
	fig = plt.figure()
	fig.set_figheight(3)
	fig.set_figwidth(7)
	gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
	ax = plt.subplot(gs[0])
	rects1 = ax.bar(ind - width/2, list_plt, width, color='royalblue')
	ax.set_ylabel('# Link Failure Combinations')
	ax.set_xlabel('Observability %')
	ax.set_xticks(ind - width/2)
	ax.set_xticklabels(obsPercent_plt)


	ax.bar_label(rects1, padding = 0.5, fontsize=8)

	#ax.legend( (rects1[0], rects2[0]), ('w/o Redundancy', 'w Redundancy') )

	#CDF
	ax2 = plt.subplot(gs[1])
	unobsList=[100-val for val in obsList]
	sns.ecdfplot(data = unobsList, color='royalblue')
	plt.ylabel(None)
	plt.xlabel("Unobservability %")
	fig.tight_layout()
	plt.show()	

def plotCDF(obsList):
	unobsList=[100-val for val in obsList]
	sns.ecdfplot(data = unobsList, color='royalblue')
	plt.ylabel(None)
	plt.xlabel("Unobservability %")
	plt.legend
	plt.show()
	
# Driver Code
if __name__ == "__main__":
	
	# PMU-Switch Network Graph using dictionaries
	graph = {'P1': ['S1'],
			'P2': ['S2'],
			'P3': ['S3'],
			'P4': ['S6'],
			'P5': ['S7'],
			'P6': ['S7'],
			'S1': ['P1','S4'],
			'S2': ['P2','S4','S7'],
			'S3': ['P3','S4','S5'],
			'S4': ['S1','S2','S3','PDC1'],
			'S5': ['S3','S6','PDC2'],
			'S6': ['P4','S5','S7'],
			'S7': ['P5','P6','S2','S6'],
			'PDC1': ['S4'],
			'PDC2': ['S5']
		}
			
	#List of all basic paths from each PMU to a PDC
	masterPathList = { 'P1' : ['S1', 'S4', 'PDC1'],
			   'P2' : ['P2', 'S2', 'S4', 'PDC1'],
			   'P4' : ['S6', 'S5', 'PDC2'],
			   'P3' : ['S3', 'S4', 'PDC1'],
			   'P5' : ['S7', 'S6', 'S5', 'PDC2'],
			   'P6' : ['S7', 'S6', 'S5', 'PDC2']
			}

	numBuses = 14	

	pmuBusMapping = { 'P1' : ['B7', 'B8', 'B9', 'B4'],
		  'P2' : ['B9', 'B7', 'B10', 'B14'],
		  'P3' : ['B2', 'B1', 'B3', 'B4', 'B5'],
		  'P4' : ['B6', 'B5', 'B11', 'B12', 'B13'],
		  'P5' : ['B6', 'B5', 'B11', 'B12', 'B13'],
		  'P6' : ['B10', 'B11', 'B9']
		}

	pdcList = ['PDC2','PDC1']

	# Dictionary to store observability for each link down combination
	observabilityDict = dict()

	linksDown = []
	numAltPaths = 0
	Count=0

	t1 = time.time()
	for node in graph:
		for i in range(len(graph[node])):

			# To include only Switch-Switch links 
			#if set([node,graph[node][i]]) not in linksDown and 'P' not in node and 'P' not in graph[node][i] :
			if set([node,graph[node][i]]) not in linksDown:

			# To include both Switch-Switch and PMU-Switch, but not PDC-Switch
			#if set([node,graph[node][i]]) not in linksDown and 'D' not in node and 'D' not in graph[node][i]:
				linksDown.append(set([node,graph[node][i]]))


	# get list of switches and initilize number of paths and degree of each switch
	switchList = [val for val in graph if 'S' in val]

	print("switchList = ", switchList)
	linksDown = [list(link) for link in linksDown]
	numLinks = len(linksDown)
	
	for switch in switchList:
		switchPathCount[switch] = []
		switchPathCount[switch].append(0) #initialize number of paths to 0
		switchPathCount[switch].append(len(graph[switch])) #append degree to respective switch

	# Initialize Num of paths per pdc to 0
	pdcList = [val for val in graph if 'PD' in val]
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
		graphTemp = copy.deepcopy(graph)

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
#	[print(key,':',value) for key, value in observabilityDict.items()]
	d_view = [ (v,k) for k,v in observabilityDict.items()]
	d_view.sort()
	for v,k in d_view:
		print("%s:\t %10.2f" % (k,v))

	print("\n# of alternate paths, degree per switch") 
	print("-----------------------------------------")
	print("Switch\tDeg\tPaths\t% Util")
	print("----------------------------")
	s_view = [ (v[0],v[1],k) for k,v in switchPathCount.items()]
	s_view.sort()
	for v0,v1,k in s_view:
		print("%s:\t%d\t%d\t%.2f" % (k,v1,v0,100*v0/numAltPaths))


	print("\n# of alternate paths per PDC") 
	print("---------------------------------")
	print("PDC\tPaths")
	print("--------------")
	[print(key + ':',value,sep='\t') for key, value in pdcPathCount.items()]

	print("\n# of alternate paths per link") 
	print("--------------------------------------")
	print("Link\t\tPaths\t% Util")
	print("--------------------------------------")
	d_view = [ (v,k) for k,v in linkPathCount.items()]
	d_view.sort()
	for v,k in d_view:
		print("%s:\t%d\t%.2f" % (k,v,100*v/numAltPaths))

	# Plot Histogram and CDF
	obsList = [val for key, val in observabilityDict.items()]

	#Bar plot for Observability values alone
	plotObs(obsList, numBuses)	
	#plotCDF(obsList)
	
	count, bins= np.histogram(obsList)
	pdf = count / sum(count)
	cdf = np.cumsum(pdf)
#	plt.figure(figsize=(20,5))

	plt.subplot(2, 2, 1)
	plt.plot(bins[1:], cdf, label="CDF")
	plt.legend()

	plt.subplot(2, 2, 2)

	#plt.hist(obsList, 100, color='g')
	plt.xlabel("Observability %")
	plt.ylabel("No. of Link Failure Combinations")
	plt.tight_layout()
	freq, bins, patches = plt.hist(obsList, edgecolor='white', color='g', label='Obs Count', bins=range(1,101,1))

	# x coordinate for labels
	bin_centers = np.diff(bins)*0.5 + bins[:-1]

	n = 0
	plt.annotate("Total Comb: %d" % (len(linksDownComb)), xy=(0,1))
	for fr, x, patch in zip(freq, bin_centers, patches):
		height = int(freq[n])
		if height!= 0:
			plt.annotate("%d (%0.2f %s)" % (height, 100 * height/len(linksDownComb), "%"), xy = (x, height),             # top left corner of the histogram bar
	       xytext = (0,0.2),             # offsetting label position above its bar
	       textcoords = "offset points", # Offset (in points) from the *xy* value
	       ha = 'center', va = 'bottom'
	       )
		n = n+1

	plt.legend()

	plt.subplot(2, 2, 3)
	linkNames = [k for v,k in d_view]
	values = [v for v,k in d_view]
	plt.bar(range(len(d_view)), values, tick_label=linkNames, width=0.8, hatch='/')#, animated=True)
	plt.xlabel("Link Combinations")
	plt.ylabel("No. of Alt. Paths")
	plt.xticks(rotation=45)
	plt.tight_layout()

	plt.subplot(2, 2, 4)
	switches = [k for v0,v1,k in s_view]
	values = [v0 for v0,v1,k in s_view]
	plt.bar(range(len(s_view)), values, tick_label=switches, width=0.8, hatch='/')#, animated=True)
	plt.xlabel("Switch")
	plt.ylabel("No. of Alt. Paths")
#	plt.xticks(rotation=45)
	plt.tight_layout()

	#plt.show()

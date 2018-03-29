import random
from parameters import *
import numpy as np
import math


def exponential_latency(avg_latency):
    """Represents the latency to transfer messages
    """
    return lambda: 1 + int(random.expovariate(1) * avg_latency)
###REGULAR GRAPH CONSTRUCTION
'''
def latency_fn(avg_latency,ADJ,source_node,target_node):
	#print(ADJ)
	path_lengths = bfs_path_len(ADJ,source_node,target_node)
	path_times = ()
	for i in path_lengths:
		#j = path_lengths[i]
		temp = 0
		temptime = 0
		while temp<i:
			temptime += 1 + int(random.expovariate(1)*avg_latency)
		path_times = path_times + (temptime,)
	return min(path_times)
'''

def cond_exp(avg_latency,x):
	if avg_latency >0:
		return 1-exp(-1/avg_latency)	
	else:
		return 1

def neighbor(ADJ,node):
	neighbor = []
	for i in range(NUM_VALIDATORS):
		if ADJ[node][i]==1:
			neighbor.append(i)
	return neighbor

def latency_fn(avg_latency,ADJ,source_node):
	delay = {}
	final_delay = np.zeros((NUM_VALIDATORS))
	for i in range(NUM_VALIDATORS):
		delay[i] = np.empty(0)
	time_count = 0
	final_delay[source_node]=time_count+1
	del delay[source_node]
	for i in neighbor(ADJ,source_node):
		delay[i] = np.append(delay[i],0)
	incomplete = True
	while incomplete:
		to_be_deleted = set()
		for i in delay:
			for j in range(len(delay[i])):
				delay[i] = delay[i] + 1
				#print (avg_latency)
				#print(delay[i][j])

				temp_prob = cond_exp(avg_latency,delay[i][j]-1)
				if np.random.uniform(0,1)< temp_prob:
					final_delay[i] = time_count +1
					to_be_deleted.add(i)
					for k in neighbor(ADJ,i):
						if k in delay:
							delay[k] = np.append(delay[k],0)
		time_count = time_count + 1
		for i in to_be_deleted:
			#print(source_node)
			del delay[i]
		if delay:
			incomplete = True
		else:
			incomplete = False
			

	return final_delay


		





'''
def bfs_path_len(ADJ,start,goal):
	#print(np.sum(ADJ,1))
	ADJ = np.array(ADJ)
	queue = [(start,[start],0)]
	abs_depth = 0

	while queue and abs_depth<6:		
		(vertex,path,depth) = queue.pop(0)
		temp_neighbors = []
		abs_depth = depth	
		for i in range(NUM_VALIDATORS):
			if ADJ[vertex][i] == 1 :
				temp_neighbors.append(i)

		print(len(queue))		
		print(abs_depth)

		for next_vertex in temp_neighbors:
			if next_vertex == goal and next_vertex not in path:
				yield len(path + [next_vertex])
				print('DEBUG2')
				input()
			elif next_vertex not in path:
				queue.append((next_vertex,path+[next_vertex],depth+1))
	print("DEBUG1")
	input()
'''
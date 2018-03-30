import random
from parameters import *


def exponential_latency (avg_latency):
    """
    Represents the latency to transfer messages
    INPUT:  avg_latency (int)
            adj_list (networkx dict of dict of dict)
            src (node.id)
    OUTPUT: list delay values for all nodes/ validators
    """
    return lambda: 1 + int(random.expovariate(1) * avg_latency)


##################################
###### for D-regular graphs ######
##################################
def generate_latencies (avg_latency, adj_list, src):
    """
    Represents the latency to transfer messages
    INPUT:  latency_fn (function)
            adj_list (networkx dict of dict of dict)
            src (node.id)
    OUTPUT: list delay values for all nodes/ validators
    """
    # number of nodes/ validators
    #n = len(adj_list)
    
    time_infected = {}
    arrival_times = [[] for i in range(NUM_VALIDATORS)]
    arrival_times[src].append(1)
    time_infected[src] = 1
    check_count = 0
    recent_infected = src

    while len(time_infected)<NUM_VALIDATORS:
        check_count += 1
        for neighbor in adj_list[recent_infected]:
            if neighbor not in time_infected:
                temp = time_infected[recent_infected] + int(random.expovariate(1)*avg_latency)
                arrival_times[neighbor].append(temp)
        temp_min = {}
        for i in range(NUM_VALIDATORS):
            if len(arrival_times[i]) and i not in time_infected:
                temp_min[i] = min(arrival_times[i])
        temp = avg_latency*NUM_VALIDATORS*1000 #random Very large number
        temp_infected = None
        for i in temp_min:
            if temp_min[i]<temp:
                temp = temp_min[i]
                temp_infected = i
        time_infected[i] = temp
        recent_infected = i

    delay_regular = [time_infected[i] for i in range(NUM_VALIDATORS)]

    #print('generating_regular_graph')

    delay_fully_connected = [exponential_latency(avg_latency) for i in range(NUM_VALIDATORS)]

    #TODO NOT IMPELMENTED

    return delay_regular

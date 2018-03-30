import random


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
def generate_latencies (latency_fn, adj_list, src):
    """
    Represents the latency to transfer messages
    INPUT:  latency_fn (function)
            adj_list (networkx dict of dict of dict)
            src (node.id)
    OUTPUT: list delay values for all nodes/ validators
    """
    # number of nodes/ validators
    n = len(adj_list)


    #TODO NOT IMPELMENTED

    return [latency_fn() for i in range(n)]

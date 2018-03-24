from parameters import *
from multiprocessing import Process

class Network(object):
    """Networking layer controlling the delivery of messages between nodes.

    self.msg_arrivals is a table where the keys are the time of arrival of
        messages and the values is a list of the objects received at that time
    """
    def __init__(self, latency_fn):
        self.nodes = []
        self.time = 0
        self.msg_arrivals = {}
        self.latency_fn = latency_fn
        self.supermajority_link = {}
        self.vote_count = {}

        self.processed = {}
        
        self.first_proposal_time = {}
        self.first_justification_time = {}
        self.global_justified_time ={}
        self.first_justified_time = {}
        self.last_justified_time = {}
        self.first_finalization_time={}
        self.global_finalized_time={}
        self.first_finalized_time={}
        self.last_finalized_time={}

        self.global_finalized_time_absolute = {}
        self.first_finalization_time_auxillary = {}
        self.first_justification_time_auxillary = {}

        self.quartile_first_finalized_time={}
        self.final_quartiles={}
        self.final_validator ={}
        self.justify_validator={}
        self.final_time={}

    def broadcast(self, msg):
        """Broadcasts a message to all nodes in the network. (with latency)

        Inputs:
            msg: the message to be broadcastes (PREPARE or COMMIT).

        Returns:
            None
        """
        for node in self.nodes:
            # Create a different delay for every receiving node i
            # Delays need to be at least 1
            delay = self.latency_fn()
            assert delay >= 1, "delay is 0, which will lose some messages !"
            if self.time + delay not in self.msg_arrivals:
                self.msg_arrivals[self.time + delay] = []
            self.msg_arrivals[self.time + delay].append((node.id, msg))

    
    def tick(self, sml_stats = {}):
        """Simulates a tick of time.

        Each node deals with receiving messages of time t.
        Increments the time of each node, and of the network.
        """
        if self.time in self.msg_arrivals:
            for node_index, msg in self.msg_arrivals[self.time]:
                self.nodes[node_index].on_receive(msg, sml_stats)
            del self.msg_arrivals[self.time]
        for n in self.nodes:
            n.tick(self.time)
        self.time += 1
        #print(self.time)
    '''
    def tick(self, sml_stats = {}):
        """Simulates a tick of time.

        Each node deals with receiving messages of time t.
        Increments the time of each node, and of the network.
        """
        P = []
        count = 0
        if self.time in self.msg_arrivals:
            for node_index, msg in self.msg_arrivals[self.time]:
                P.append(Process(target =self.nodes[node_index].on_receive(msg, sml_stats)))
                P[count].start()
                count = count + 1
            for i in range(0,count):
                P[i].join()
            del self.msg_arrivals[self.time]
        for n in self.nodes:
            n.tick(self.time)
        self.time += 1
        #print(self.time)
    '''
     
    def report_proposal(self,block):
        self.first_proposal_time[block.hash] = self.time
        self.processed[block.hash] = block

    def report_vote(self,vote):
        if vote.source not in self.vote_count:
            self.vote_count[vote.source] = {}
        self.vote_count[vote.source][vote.target] = self.vote_count[vote.source].get(vote.target,0) + 1
        if self.vote_count[vote.source][vote.target] ==1:
            if vote.source not in self.first_justification_time_auxillary:
                self.first_justification_time_auxillary[vote.source] = {}
            self.first_justification_time_auxillary[vote.source][vote.target] = self.time - self.first_proposal_time[vote.target]
            if vote.epoch_target - vote.epoch_source == 1:
                if vote.source not in self.first_finalization_time_auxillary:
                    self.first_finalization_time_auxillary[vote.source] = {}
                self.first_finalization_time_auxillary[vote.source][vote.target] = self.time - self.first_proposal_time[vote.source]
        if self.vote_count[vote.source][vote.target] == (NUM_VALIDATORS*1)//2 + 1:
            if vote.source not in self.supermajority_link:
                self.supermajority_link[vote.source] = []           ##In this function we assume all validators are honest, hence no checking condition
            self.supermajority_link[vote.source].append(vote.target) # if there is a vote from vote.source, it mean it was justified
            self.global_justified_time[vote.target] = self.time - self.first_proposal_time[vote.target]
            self.first_justification_time[vote.target] = self.first_justification_time_auxillary[vote.source][vote.target]
            if vote.epoch_target - vote.epoch_source ==1 :
                self.global_finalized_time[vote.source] = self.time - self.first_proposal_time[vote.source]
                self.global_finalized_time_absolute[vote.source] = self.time
                prev_block_hash = self.processed[vote.source].prev_hash
                while prev_block_hash !=0 and prev_block_hash not in self.global_finalized_time_absolute:
                    self.global_finalized_time_absolute[prev_block_hash] = self.time
                    prev_block_hash = self.processed[prev_block_hash].prev_hash

                self.first_finalization_time[vote.source] = self.first_finalization_time_auxillary[vote.source][vote.target]

    def report_justified(self,blockhash,val_id):
        if blockhash not in self.justify_validator:
            self.justify_validator[blockhash] = []
            self.first_justified_time[blockhash] = self.time - self.first_proposal_time[blockhash]
        self.justify_validator[blockhash].append(val_id)
        if len(self.justify_validator[blockhash])==NUM_VALIDATORS:
            self.last_justified_time[blockhash] = self.time - self.first_proposal_time[blockhash]



    def report_finalized(self,blockhash,val_id):
        ##Assumed all honest validators here, no multiple finalizations published, can edit it to check honesty
        if blockhash not in self.final_validator:
            self.final_validator[blockhash] = []
            self.first_finalized_time[blockhash] = self.time - self.first_proposal_time[blockhash]
        self.final_validator[blockhash].append(val_id)
        if len(self.final_validator[blockhash])==NUM_VALIDATORS:
            self.last_finalized_time[blockhash] = self.time - self.first_proposal_time[blockhash]
        if len(self.final_validator[blockhash])>=NUM_VALIDATORS//2:
            self.final_time[blockhash] = self.time
        if blockhash not in self.final_quartiles:
            self.final_quartiles[blockhash] = []
            self.quartile_first_finalized_time[blockhash] = self.time

        temp = NUM_VALIDATORS//4
        if len(self.final_validator[blockhash])%temp == 0 :
            self.final_quartiles[blockhash].append(self.time-self.quartile_first_finalized_time[blockhash])




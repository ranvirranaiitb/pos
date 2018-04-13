from block import Block, Dynasty
from message import Vote
from parameters import *
import numpy as np
import random
import math

# Root of the blockchain
ROOT = Block()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class Validator(object):
    """Abstract class for validators."""

    def __init__(self, network,latency, wait_fraction, id, immediate_vote, random_proposal_wait):
        # processed blocks
        self.processed = {ROOT.hash: ROOT}
        # Messages that are not processed yet, and require another message
        # to be processed
        # Dict from hash of dependency to object that can be processed
        # when dependency is processed
        # Example:
        # prepare messages processed before block is processed
        # commit messages processed before we reached 2/3 prepares
        self.dependencies = {}
        self.justification_dependencies = {}
        # Set of finalized dynasties
        self.finalized_dynasties = set()
        self.finalized_dynasties.add(Dynasty(INITIAL_VALIDATORS))
        # My current epoch
        self.current_epoch = 0
        # Network I am connected to
        self.network = network
        network.nodes.append(self)
        # Tails are for checkpoint blocks, the tail is the last block
        # (before the next checkpoint) following the checkpoint
        self.tails = {ROOT.hash: ROOT}
        # Closest checkpoint ancestor for each block
        self.tail_membership = {ROOT.hash: ROOT.hash}
        self.id = id
        self.network.report_proposal(ROOT)
        self.blocks_received = {ROOT.hash}

        #self.latency = latency
        #self.wait_fraction = wait_fraction

        self.voting_delay_average = latency*wait_fraction
        self.mining_id = id             #Mining id, shuffeled after every proposal
        self.immediate_vote = immediate_vote
        self.random_proposal_wait = random_proposal_wait
        self.block_proposal_permission = False
        self.block_proposal_wait = None
        self.latest_received_block_time = 0
        self.pseudo_head = ROOT
        self.bpm_check1 = 0
        self.bpm_check2 = 0

    # If we processed an object but did not receive some dependencies
    # needed to process it, save it to be processed later
    def add_dependency(self, hash_, obj):
        if hash_ not in self.dependencies:
            self.dependencies[hash_] = []
        self.dependencies[hash_].append(obj)

    def add_justification_dependency(self,hash_,obj):
        if hash_ not in self.justification_dependencies:
            self.justification_dependencies[hash_] = []
        self.justification_dependencies[hash_].append(obj)

    # Get the checkpoint immediately before a given checkpoint
    def get_checkpoint_parent(self, block):
        if block.height == 0:
            return None
        return self.processed[self.tail_membership[block.prev_hash]]

    def is_ancestor(self, anc, desc):
        """Is a given checkpoint an ancestor of another given checkpoint?
        Args:
            anc: ancestor block (or block hash)
            desc: descendant block (or block hash)
        """
        # if anc or desc are block hashes, we can get the blocks from self.processed
        # TODO: but what if they are not in processed? BUG?
        if not isinstance(anc, Block):
            anc = self.processed[anc]
        if not isinstance(desc, Block):
            desc = self.processed[desc]
        # Check that the blocks are both checkpoints
        assert anc.height % EPOCH_SIZE == 0
        assert desc.height % EPOCH_SIZE == 0
        while desc.height >= anc.height:
            if desc is None:
                return False
            if desc.hash == anc.hash:
                return True
            desc = self.get_checkpoint_parent(desc)
        return False

    def make_tree(self,block_dict):
        block_tree = {}
        for block_hash in block_dict:
            if block_dict[block_hash][0].height not in block_tree:
                block_tree[block_dict[block_hash][0].height] = {}
            block_tree[block_dict[block_hash][0].height][block_hash] = block_dict[block_hash]
        
        return block_tree

    def score_blocks(self,block_dict):
        for block_hash in block_dict:
            prev_hash = block_dict[block_hash][0].prev_hash
            if prev_hash in block_dict:
                temp_block = block_dict[block_dict[block_hash][0].prev_hash]
                score_counter = block_dict[block_hash][1]
                while temp_block[0].height > self.highest_justified_checkpoint.height:
                    score_counter += temp_block[1]
                    prev_hash = temp_block[0].prev_hash
                    if prev_hash not in block_dict:
                        score_counter = 0
                        break
                    else:
                        temp_block = block_dict[prev_hash]
                if not temp_block[0].hash == self.highest_justified_checkpoint.hash:
                    score_counter = 0
            else:
                score_counter = 0
            block_dict[block_hash].append(score_counter)

        return block_dict

    def return_polled_pseudo_head(self,block_dict):
        temp_score = 0
        temp_hash = None
        for block_hash in block_dict:
            if block_dict[block_hash][2] > temp_score:
                temp_score = block_dict[block_hash][2]
                temp_hash = block_hash
        if temp_score == 0:
            self.bpm_check1 += 1
            return self.head
        else:
            self.bpm_check2 += 1
            return block_dict[temp_hash][0]




    def join_block_dict(self,block_dict_1, block_dict_2):
        temp_dict = block_dict_1
        for key in block_dict_2:
            if key not in temp_dict:
                temp_dict[key] = block_dict_2[key]
            else:
                temp_dict[key][1] += block_dict_2[key][1]

        return temp_dict 

    # Called every round
    def tick(self, time, sml_stats = {}):
        # At time 0: validator 0
        # At time BLOCK_PROPOSAL_TIME: validator 1
        # .. At time NUM_VALIDATORS * BLOCK_PROPOSAL_TIME: validator 0
        if not self.immediate_vote:
            self.vote_on_delay()

        '''    
        if self.mining_id == (time // BLOCK_PROPOSAL_TIME) % NUM_VALIDATORS :
            if not self.block_proposal_permission :
                self. block_proposal_wait = int(np.random.uniform(0,BLOCK_PROPOSAL_TIME*self.random_proposal_wait))
                self.block_proposal_permission = True
        '''
        '''
        self.block_proposal_wait = 0
        if (self.mining_id+1) == (time // BLOCK_PROPOSAL_TIME) % NUM_VALIDATORS :
            self.block_proposal_permission = True

        if self.mining_id == (time // BLOCK_PROPOSAL_TIME) % NUM_VALIDATORS:
            time_diff = time - self.block_proposal_wait - self.latest_received_block_time
            if time%BLOCK_PROPOSAL_TIME > BLOCK_PROPOSAL_TIME-2 or time_diff ==1 :
                if self.block_proposal_permission:            
                    # One node is authorized to create a new block and broadcast it
                    new_block = Block(self.head, self.finalized_dynasties)
                    self.network.broadcast(new_block, self.id)
                    self.network.report_proposal(new_block)
                    self.on_receive(new_block, sml_stats)  # immediately "receive" the new block (no network latency)
                    self.block_proposal_permission = False

        '''


        if self.mining_id == (time // BLOCK_PROPOSAL_TIME) % NUM_VALIDATORS and time % BLOCK_PROPOSAL_TIME == 0:
            # One node is authorized to create a new block and broadcast it
            '''
            temp_head = self.pseudo_head
            temp_head_height = self.pseudo_head.height
            p_communication = 10.0/NUM_VALIDATORS
            for node in self.network.nodes:
                if np.random.binomial(1,p_communication):
                    if node.pseudo_head.height > temp_head_height:
                        temp_head_height = node.pseudo_head.height
                        temp_head = node.pseudo_head
            
            '''
            '''
            temp_head = self.head
            temp_head_height = self.head.height
            p_communication = 10.0/NUM_VALIDATORS
            for node in self.network.nodes:
                if np.random.binomial(1,p_communication):
                    if node.head.height > temp_head_height:
                        temp_head_height = node.head.height
                        temp_head = node.head
            '''
            
            if self.bpm ==1:
                min_height = self.highest_justified_checkpoint.height
                block_dict = self.send_top_tree(min_height)
                p_communication = 4.0/NUM_VALIDATORS
                for node in self.network.nodes:
                    if np.random.binomial(1,p_communication):
                        temp_dict = node.send_top_tree(min_height)
                        block_dict = self.join_block_dict(block_dict, temp_dict)

                block_dict = self.score_blocks(block_dict)
                temp_head = self.return_polled_pseudo_head(block_dict)
                #block_tree = self.make_tree(block_dict)

            if self.bpm ==0:
                temp_head = self.head


            #print(temp_head_height)
            new_block = Block(temp_head, self.finalized_dynasties)
            self.network.broadcast(new_block, self.id)
            self.network.report_proposal(new_block)
            self.on_receive(new_block, sml_stats)  

class VoteValidator(Validator):
    """Add the vote messages + slashing conditions capability"""

    def __init__(self, network, latency, wait_fraction, id, vote_as_block, immediate_vote, wait_for_majority, vote_confidence, lottery_fraction, random_proposal_wait, bpm):
        super(VoteValidator, self).__init__(network, latency,wait_fraction, id,immediate_vote, random_proposal_wait)
        # the head is the latest block processed descendant of the highest
        # justified checkpoint
        self.head = ROOT
        self.highest_justified_checkpoint = ROOT
        self.main_chain_size = 0

        # Set of justified block hashes
        self.justified = {ROOT.hash}

        # Set of finalized block hashes
        self.finalized = {ROOT.hash}

        # Map {sender -> votes}
        # Contains all the votes, and allow us to see who voted for whom
        # Used to check for the slashing conditions
        self.votes = {}

        # Map {source_hash -> {target_hash -> count}} to count the votes
        # ex: self.vote_count[source][target] will be between 0 and NUM_VALIDATORS
        self.vote_count = {}

        self.depth_finalized = 0
        self.num_depth_finalized = 0
        self.highest_finalized_checkpoint_epoch = 0

        self.time_to_vote = {} # Height: int
        self.vote_permission = {} #Height:bool
        self.first_block_height={} #Height:block
        self.type_1_vote = 0
        self.type_2_vote = 0
        self.vote_as_block = vote_as_block
        self.wait_for_majority = wait_for_majority
        self.vote_confidence = vote_confidence
        self.lottery_fraction = lottery_fraction
        self.vote_score = {}
        self.num_votes = 0
        self.vote_target_count = {}
        self.bpm = bpm
        

    # TODO: we could write function is_justified only based on self.processed and self.votes
    #       (note that the votes are also stored in self.processed)
    def is_justified(self, _hash):
        """Returns True if the `_hash` corresponds to a justified checkpoint.

        A checkpoint c is justified if there exists a supermajority link (c' -> c) where
        c' is justified. The genesis block is justified.
        """
        # Check that the function is called only on checkpoints
        assert _hash in self.processed, "Couldn't find block hash %d" % _hash
        assert self.processed[_hash].height % EPOCH_SIZE == 0, "Block is not a checkpoint"

        return _hash in self.justified

    def is_finalized(self, _hash):
        """Returns True if the `_hash` corresponds to a justified checkpoint.

        A checkpoint c is justified if there exists a supermajority link (c' -> c) where
        c' is justified. The genesis block is justified.
        """
        # Check that the function is called only on checkpoints
        assert _hash in self.processed, "Couldn't find block hash %d" % _hash
        assert self.processed[_hash].height % EPOCH_SIZE == 0, "Block is not a checkpoint"

        return _hash in self.finalized

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        self._head = value

    def accept_block(self, block):
        """Called on receiving a block

        Args:
            block: block processed

        Returns:
            True if block was accepted or False if we are missing dependencies
        """

        self.blocks_received.add(block.hash)

        # If we didn't receive the block's parent yet, wait
        if block.prev_hash not in self.processed:
            self.add_dependency(block.prev_hash, block)
            return False

        # We receive the block
        self.processed[block.hash] = block

        self.depth_finalized += block.height - self.highest_finalized_checkpoint_epoch*EPOCH_SIZE
        self.num_depth_finalized += 1

        if block.height not in self.first_block_height:
            self.first_block_height[block.height] = block

        # If it's an epoch block (in general)
        if block.height % EPOCH_SIZE == 0:
            #  Start a tail object for it
            self.tail_membership[block.hash] = block.hash
            self.tails[block.hash] = block
            # Maybe vote
            if block.height not in self.time_to_vote:
                toss = np.random.binomial(1,self.lottery_fraction)
                if toss:
                    self.time_to_vote[block.height] = self.network.time + 1
                    self.vote_permission[block.height] = False 
                else:
                    self.time_to_vote[block.height] = self.network.time + 1 + int(random.expovariate(1) * self.voting_delay_average)        
                    self.vote_permission[block.height] = False
                

            if self.immediate_vote:
                self.maybe_vote_last_checkpoint(block,0.5)

        # Otherwise...
        else:
            # See if it's part of the longest tail, if so set the tail accordingly
            assert block.prev_hash in self.tail_membership
            # The new block is in the same tail as its parent
            self.tail_membership[block.hash] = self.tail_membership[block.prev_hash]
            # If the block has the highest height, it becomes the end of the tail
            if block.height > self.tails[self.tail_membership[block.hash]].height:
                self.tails[self.tail_membership[block.hash]] = block

        # Reorganize the head
        self.check_head(block)
        return True

    def check_head(self, block):
        """Reorganize the head to stay on the chain with the highest
        justified checkpoint.

        If we are on wrong chain, reset the head to be the highest descendent
        among the chains containing the highest justified checkpoint.

        Args:
            block: latest block processed."""

        # we are on the right chain, the head is simply the latest block
        if self.is_ancestor(self.highest_justified_checkpoint,
                            self.tail_membership[block.hash]):
        	if self.head.height < block.height:
        		self.head = block
        		self.main_chain_size = block.height

        # otherwise, we are not on the right chain
        else:
            # Find the highest descendant of the highest justified checkpoint
            # and set it as head
            # print('Wrong chain, reset the chain to be a descendant of the '
                  # 'highest justified checkpoint.')
            max_height = self.highest_justified_checkpoint.height
            max_descendant = self.highest_justified_checkpoint.hash
            for _hash in self.tails:
                # if the tail is descendant to the highest justified checkpoint
                # TODO: bug with is_ancestor? see higher
                if self.is_ancestor(self.highest_justified_checkpoint, _hash):
                    new_height = self.processed[_hash].height
                    if new_height > max_height:
                        max_height = new_height
                        max_descendant = _hash

            self.main_chain_size = max_height
            self.head = self.processed[max_descendant]

    def maybe_vote_last_checkpoint(self, block, confidence):
        """Called after receiving a block.

        Implement the fork rule:
        maybe send a vote message where target is block
        if we are on the chain containing the justified checkpoint of the
        highest height, and we have never sent a vote for this height.

        Args:
            block: last block we processed
        """
        assert block.height % EPOCH_SIZE == 0, (
            "Block {} is not a checkpoint.".format(block.hash))

        # BNO: The target will be block (which is a checkpoint)
        target_block = block
        # BNO: The source will be the justified checkpoint of greatest height
        source_block = self.highest_justified_checkpoint

        #print('DEBUG6')
        #input()

        # If the block is an epoch block of a higher epoch than what we've seen so far
        # This means that it's the first time we see a checkpoint at this height
        # It also means we never voted for any other checkpoint at this height (rule 1)
        if target_block.epoch > self.current_epoch:
            #print(target_block.epoch - self.current_epoch)
            #assert target_block.epoch > source_block.epoch, ("target epoch: {},"
            #"source epoch: {}".format(target_block.epoch, source_block.epoch))

            # print('Validator %d: now in epoch %d' % (self.id, target_block.epoch))
            # Increment our epoch
            if target_block.epoch>source_block.epoch:
                #self.current_epoch = target_block.epoch

                # if the target_block is a descendent of the source_block, send
                # a vote
                if self.is_ancestor(source_block, target_block):
                    self.current_epoch = target_block.epoch
                    # print('Validator %d: Voting %d for epoch %d with epoch source %d' %
                          # (self.id, target_block.hash, target_block.epoch,
                           # source_block.epoch))

                    vote = Vote(source_block.hash,
                                target_block.hash,
                                source_block.epoch,
                                target_block.epoch,
                                self.id, confidence)
                    self.network.broadcast(vote, self.id)
                    self.network.report_vote(vote)
                    #print('Debug_vote_1')
                    self.num_votes+=1

                    del self.time_to_vote[block.height]
                    del self.vote_permission[block.height]
                    '''
                    print('DEBUG5')
                    print(source_block.hash)
                    print(target_block.hash)
                    print(source_block.epoch)
                    print(target_block.epoch)
                    #input()
                    '''
                    assert self.processed[target_block.hash]

                    return True
                else:
                    return False
            else:
                return False
        else:
            return False



    def accept_vote(self, vote, sml_stats = {}):
        """Called on receiving a vote message.
        """
        # print('Node %d: got a vote' % self.id, source.view, prepare.view_source,
              # prepare.blockhash, vote.blockhash in self.processed)
        
        '''
       # If the block has not yet been processed, wait
        if vote.source not in self.processed:
            self.add_dependency(vote.source, vote)

        # Check that the source is processed and justified
        # TODO: If the source is not justified, add to dependencies?
        #******************************ADD DEPENDENCIES HERE************************************

        if vote.source not in self.justified:
            self.add_justification_dependency(vote.source,vote)  ##########
            return False
        # If the target has not yet been processed, wait
        if vote.target not in self.processed:
            self.add_dependency(vote.target, vote)
            return False
        '''

        # If the target is not a descendent of the source, ignore the vote
        '''
        if not self.is_ancestor(vote.source, vote.target):
            return False
        '''

        # If the sender is not in the block's dynasty, ignore the vote
        # TODO: is it really vote.target? (to check dynasties)
        # TODO: reorganize dynasties like the paper
        #*****************Dynasty management not implemented************************************
        '''
        if vote.sender not in self.processed[vote.target].current_dynasty.validators and \
            vote.sender not in self.processed[vote.target].prev_dynasty.validators:
            return False
        '''

        # Initialize self.votes[vote.sender] if necessary
        '''
        print('vote received')
        print(self.network.time)
        print('vote.epoch_target: {}'.format(vote.epoch_target))
        print(len(self.blocks_received))
        input()
        '''
        
        #print(self.vote_as_block)

        if self.vote_as_block:
            temphash = vote.source
            hashlist = []
            for i in range(EPOCH_SIZE):
                #print(i)
                if temphash not in self.blocks_received:
                    hashlist.append(temphash)
                    temphash = self.network.processed[temphash].prev_hash
            hashlist.reverse()
            for blockhash in hashlist:
                self.on_receive(self.network.processed[blockhash],sml_stats)
                #print('Block received via vote source')
                #input()
            temphash = vote.target
            hashlist = []
            for i in range(EPOCH_SIZE):
                if temphash not in self.blocks_received:
                    hashlist.append(temphash)
                    temphash = self.network.processed[temphash].prev_hash
            hashlist.reverse()
            for blockhash in hashlist:
                self.on_receive(self.network.processed[blockhash],sml_stats)
                #print('Block received via vote target')
                #input()

        if vote.sender not in self.votes:
            self.votes[vote.sender] = []



        # Add the vote to the map of votes
        
        if vote not in self.votes[vote.sender]:

            # Check the slashing conditions
            for past_vote in self.votes[vote.sender]:
                if past_vote.epoch_target == vote.epoch_target:
                    # TODO: SLASH

                    print('You just got slashed')
                    return False

                if ((past_vote.epoch_source < vote.epoch_source and
                     past_vote.epoch_target > vote.epoch_target) or
                   (past_vote.epoch_source > vote.epoch_source and
                     past_vote.epoch_target < vote.epoch_target)):
                    print('You just got slashed.')
                    return False

            self.votes[vote.sender].append(vote)

            # Add to the vote count
            if vote.source not in self.vote_count:
                self.vote_count[vote.source] = {}
            self.vote_count[vote.source][vote.target] = self.vote_count[
                vote.source].get(vote.target, 0) + 1      

            if vote.source not in self.vote_score:
                self.vote_score[vote.source] = {}
            self.vote_score[vote.source][vote.target] = self.vote_score[
                vote.source].get(vote.target, 0) + vote.confidence   

            self.vote_target_count[vote.target] = self.vote_target_count.get(vote.target,0) + 1        

            

        # TODO: we do not deal with finalized dynasties (the pool of validator
        # is always the same right now)
        # If there are enough votes, process them
        
        accept_vote_return = self.check_SM(vote,sml_stats)

        return accept_vote_return
        


    def check_SM(self,vote,sml_stats):
        #print('Checking SM')

        if vote.source not in self.processed:
            self.add_dependency(vote.source, vote)
            return False

        # Check that the source is processed and justified
        # TODO: If the source is not justified, add to dependencies?
        #******************************ADD DEPENDENCIES HERE************************************

        if vote.source not in self.justified:
            self.add_justification_dependency(vote.source,vote)  ##########
            return False
        # If the target has not yet been processed, wait
        if vote.target not in self.processed:
            self.add_dependency(vote.target, vote)
            return False



        if vote.target in self.processed and vote.source in self.processed:
            #print('Checking processed')    

            if (self.vote_count[vote.source][vote.target] > (NUM_VALIDATORS * SUPER_MAJORITY)):
                #print('SM')
                # record the length of a link
                sml_stats[(vote.source, vote.target)] = vote.epoch_target - vote.epoch_source

                # Mark the target as justified
                self.justified.add(vote.target)
                self.network.report_justified(vote.target,self.id)


                if vote.target in self.justification_dependencies:
                    for d in self.justification_dependencies[vote.target]:
                        self.on_receive(d,sml_stats)
                    del self.justification_dependencies[vote.target]

                if vote.epoch_target > self.highest_justified_checkpoint.epoch:
                    self.highest_justified_checkpoint = self.processed[vote.target]
                    #self.time_to_vote = self.network.time + np.random.uniform(0,VOTING_DELAY)
        
                # If the source was a direct parent of the target, the source
                # is finalized
                if vote.epoch_source == vote.epoch_target - 1:
                    if vote.epoch_source>self.highest_finalized_checkpoint_epoch:
                        self.highest_finalized_checkpoint_epoch = vote.epoch_source
                    self.finalized.add(vote.source)
                    self.network.report_finalized(vote.source,self.id)
        
        return True    




    '''
    def vote_if_delayed(self):
        if self.network.time >= self.time_to_vote:
            temp = 0
            temptarget = None
            for targethash in self.vote_count[self.highest_justified_checkpoint.hash]:
                if self.vote_count[self.highest_justified_checkpoint.hash][targethash]>temp:
                    temp = self.vote_count[self.highest_justified_checkpoint.hash][targethash]
                    temptarget = targethash
            if temp>0:
                if temptarget in self.processed:
                    self.maybe_vote_last_checkpoint(self.processed[temptarget])
            else:
                self.vote_permission = True
    '''

    def vote_on_delay(self):
        for blockheight in range((self.current_epoch+1)*EPOCH_SIZE,self.head.height,EPOCH_SIZE):
            if blockheight in self.time_to_vote:
                if self.network.time > self.time_to_vote[blockheight] :
                    if self.highest_justified_checkpoint.hash not in self.vote_score:
                        self.vote_score[self.highest_justified_checkpoint.hash] = {}
                    temp_vote_score = self.vote_score[self.highest_justified_checkpoint.hash]
                    self.vote_at_given_height(blockheight,temp_vote_score )
                    '''
                    print("I am here-2")
                    print(self.current_epoch)
                    input()
                    '''


    def vote_at_given_height(self,blockheight, temp_vote_score):
        temp = 0
        temptarget = None

        for targethash in temp_vote_score:
            if self.vote_score[self.highest_justified_checkpoint.hash][targethash]>temp and self.network.processed[targethash].height == blockheight :
                temp = self.vote_score[self.highest_justified_checkpoint.hash][targethash]
                temptarget = targethash
        
        if temp>0:

            if self.vote_confidence:
                v1 = self.vote_score[self.highest_justified_checkpoint.hash][temptarget]    #Doubt, how about vote_score or vote_count
                sum_v = 0
                for targethash in self.vote_score[self.highest_justified_checkpoint.hash]:
                    sum_v += self.vote_score[self.highest_justified_checkpoint.hash][targethash]
                p_confidence = v1/sum_v
                #print(p_confidence)
                sum_v_count = 0
                for targethash in self.vote_count[self.highest_justified_checkpoint.hash]:
                    sum_v_count += self.vote_count[self.highest_justified_checkpoint.hash][targethash]

                num_confidence = 4.5*sum_v_count/NUM_VALIDATORS + 0.5

                confidence = p_confidence*num_confidence
            else:
                confidence = 0.5


            if temptarget in self.processed:
                #print("I am here")
                #input()
                #print(self.processed[temptarget].height)
                #print(self.current_epoch)
                #print('This is happening')
                self.type_1_vote += self.maybe_vote_last_checkpoint(self.processed[temptarget], confidence) #If the targetblock has not yet arrived, wait till the targetblock arrives and then do the maximization
            else:
                if not self.wait_for_majority:
                    del temp_vote_score[temptarget]
                    self.vote_at_given_height(blockheight,temp_vote_score)
        else:
            self.vote_permission[blockheight] = True
            #print(blockheight)
            #print(self.current_epoch)

            self.type_2_vote += self.maybe_vote_last_checkpoint(self.first_block_height[blockheight], 0.5)

            #print('DEBUG4')
            #input()

    def check_block_validity(self,block):
        #if block.hash in self.blocks_received:
        #    return False

        if block.height > self.highest_finalized_checkpoint_epoch*EPOCH_SIZE:
            return True
        else:
            return False

    def check_vote_validity(self,vote):
        if vote.epoch_target < self.highest_finalized_checkpoint_epoch :
            return False
        else:
            return True

    def send_top_tree(self,min_height):
        return_dict = {}
        for block_hash in self.blocks_received:
            if self.network.processed[block_hash].height>=min_height:
                temp = self.vote_target_count.get(block_hash,1)
                return_dict[block_hash] = [self.network.processed[block_hash],temp]
        return return_dict


    # Called on processing any object
    def on_receive(self, obj, sml_stats={}):
        if obj.hash in self.processed:
            return False
        if isinstance(obj, Block):
            val = self.check_block_validity(obj)
            if obj.height > self.pseudo_head.height:
                #print(obj.height)
                self.pseudo_head = obj
            if val:
                o = self.accept_block(obj)
                self.latest_received_block_time = self.network.time ##DECISION: WHERE TO PLACE THIS, this location is worth scrutinys
                #print('Block on receive called')
        elif isinstance(obj, Vote):
            val = self.check_vote_validity(obj)
            if val:
                o = self.accept_vote(obj, sml_stats)
        # If the object was successfully processed
        # (ie. not flagged as having unsatisfied dependencies)
        if val:
            if o:
                self.processed[obj.hash] = obj
                if obj.hash in self.dependencies:
                    for d in self.dependencies[obj.hash]:
                        self.on_receive(d,sml_stats)
                    del self.dependencies[obj.hash]

        if not val:
            if obj.hash in self.dependencies:
                del self.dependencies[obj.hash]

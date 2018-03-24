from block import Block, Dynasty
from message import Vote
from parameters import *
import numpy as np
import random

# Root of the blockchain
ROOT = Block()

class Validator(object):
    """Abstract class for validators."""

    def __init__(self, network,latency, wait_fraction, id):
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

        #self.latency = latency
        #self.wait_fraction = wait_fraction

        self.voting_delay_average = latency*wait_fraction

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
        while True:
            if desc is None:
                return False
            if desc.hash == anc.hash:
                return True
            desc = self.get_checkpoint_parent(desc)

    # Called every round
    def tick(self, time, sml_stats = {}):
        # At time 0: validator 0
        # At time BLOCK_PROPOSAL_TIME: validator 1
        # .. At time NUM_VALIDATORS * BLOCK_PROPOSAL_TIME: validator 0
        self.vote_on_delay()
        if self.id == (time // BLOCK_PROPOSAL_TIME) % NUM_VALIDATORS and time % BLOCK_PROPOSAL_TIME == 0:
            # One node is authorized to create a new block and broadcast it
            new_block = Block(self.head, self.finalized_dynasties)
            self.network.broadcast(new_block)
            self.network.report_proposal(new_block)
            self.on_receive(new_block, sml_stats)  # immediately "receive" the new block (no network latency)

class VoteValidator(Validator):
    """Add the vote messages + slashing conditions capability"""

    def __init__(self, network, latency, wait_fraction, id):
        super(VoteValidator, self).__init__(network, latency,wait_fraction, id)
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
                self.time_to_vote[block.height] = self.network.time + 1 + int(random.expovariate(1) * self.voting_delay_average)
                self.vote_permission[block.height] = False

            #if self.vote_permission.get(block.height,False):
            #    self.maybe_vote_last_checkpoint(block)

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

    def maybe_vote_last_checkpoint(self, block):
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
            #assert target_block.epoch > source_block.epoch, ("target epoch: {},"
            #"source epoch: {}".format(target_block.epoch, source_block.epoch))

            # print('Validator %d: now in epoch %d' % (self.id, target_block.epoch))
            # Increment our epoch
            if target_block.epoch>source_block.epoch:
                self.current_epoch = target_block.epoch

                # if the target_block is a descendent of the source_block, send
                # a vote
                if self.is_ancestor(source_block, target_block):
                    # print('Validator %d: Voting %d for epoch %d with epoch source %d' %
                          # (self.id, target_block.hash, target_block.epoch,
                           # source_block.epoch))

                    vote = Vote(source_block.hash,
                                target_block.hash,
                                source_block.epoch,
                                target_block.epoch,
                                self.id)
                    self.network.broadcast(vote)
                    self.network.report_vote(vote)

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

    def accept_vote(self, vote, sml_stats = {}):
        """Called on receiving a vote message.
        """
        # print('Node %d: got a vote' % self.id, source.view, prepare.view_source,
              # prepare.blockhash, vote.blockhash in self.processed)

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

        # If the target is not a descendent of the source, ignore the vote
        if not self.is_ancestor(vote.source, vote.target):
            return False

        # If the sender is not in the block's dynasty, ignore the vote
        # TODO: is it really vote.target? (to check dynasties)
        # TODO: reorganize dynasties like the paper
        #*****************Dynasty management not implemented************************************

        if vote.sender not in self.processed[vote.target].current_dynasty.validators and \
            vote.sender not in self.processed[vote.target].prev_dynasty.validators:
            return False

        # Initialize self.votes[vote.sender] if necessary
        if vote.sender not in self.votes:
            self.votes[vote.sender] = []

        # Check the slashing conditions
        for past_vote in self.votes[vote.sender]:
            if past_vote.epoch_target == vote.epoch_target:
                # TODO: SLASH
                print('You just got slashed.')
                return False

            if ((past_vote.epoch_source < vote.epoch_source and
                 past_vote.epoch_target > vote.epoch_target) or
               (past_vote.epoch_source > vote.epoch_source and
                 past_vote.epoch_target < vote.epoch_target)):
                print('You just got slashed.')
                return False

        # Add the vote to the map of votes
        self.votes[vote.sender].append(vote)

        # Add to the vote count
        if vote.source not in self.vote_count:
            self.vote_count[vote.source] = {}
        self.vote_count[vote.source][vote.target] = self.vote_count[
            vote.source].get(vote.target, 0) + 1

        # TODO: we do not deal with finalized dynasties (the pool of validator
        # is always the same right now)
        # If there are enough votes, process them
        if (self.vote_count[vote.source][vote.target] > (NUM_VALIDATORS * SUPER_MAJORITY)):

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
                    self.vote_at_given_height(blockheight)
                    '''
                    print("I am here-2")
                    print(self.current_epoch)
                    input()
                    '''

    def vote_at_given_height(self,blockheight):
        temp = 0
        temptarget = None
        if self.highest_justified_checkpoint.hash not in self.vote_count:
            self.vote_count[self.highest_justified_checkpoint.hash] = {}
        for targethash in self.vote_count[self.highest_justified_checkpoint.hash]:
            if self.vote_count[self.highest_justified_checkpoint.hash][targethash]>temp and self.network.processed[targethash].height == blockheight :
                temp = self.vote_count[self.highest_justified_checkpoint.hash][targethash]
                temptarget = targethash
        if temp>0:
            #print("DEBUG3")
            #input()
            if temptarget in self.processed:
                #print("I am here")
                #input()
                #print(self.processed[temptarget].height)
                #print(self.current_epoch)
                self.maybe_vote_last_checkpoint(self.processed[temptarget]) #If the targetblock has not yet arrived, wait till the targetblock arrives and then do the maximization
        else:
            self.vote_permission[blockheight] = True
            #print(blockheight)
            #print(self.current_epoch)

            self.maybe_vote_last_checkpoint(self.first_block_height[blockheight])

            #print('DEBUG4')
            #input()

    def check_block_validity(self,block):
        if block.height > self.highest_finalized_checkpoint_epoch*EPOCH_SIZE:
            return True
        else:
            return False

    def check_vote_validity(self,vote):
        if vote.epoch_target < self.highest_finalized_checkpoint_epoch :
            return False
        else:
            return True

    # Called on processing any object
    def on_receive(self, obj, sml_stats={}):
        if obj.hash in self.processed:
            return False
        if isinstance(obj, Block):
            val = self.check_block_validity(obj)
            if val:
                o = self.accept_block(obj)
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

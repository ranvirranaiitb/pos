"""List of parameters for the simulation.
"""

NUM_VALIDATORS = 500  # number of validators at each checkpoint
VALIDATOR_IDS = list(range(0, NUM_VALIDATORS * 2))  # set of validators
INITIAL_VALIDATORS = list(range(0, NUM_VALIDATORS))  # set of validators for root
BLOCK_PROPOSAL_TIME = 100  # adds a block every 100 ticks
EPOCH_SIZE = 5  # checkpoint every 5 blocks
AVG_LATENCY = 10  # average latency of the network (in number of ticks)

NUM_EPOCH = 2500
#VOTING_DELAY = 2*BLOCK_PROPOSAL_TIME+1
SUPER_MAJORITY = 0.67

# graph properties
D_REGULAR = 4

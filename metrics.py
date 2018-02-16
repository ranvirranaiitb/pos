import os
import numpy as np

from parameters import *
from block import Block
from utils import exponential_latency
from network import Network
from validator import VoteValidator
from plot_graph import plot_node_blockchains


def fraction_justified_and_finalized(validator):
    """Compute the fraction of justified and finalized checkpoints in the main chain.

    From the genesis block to the highest justified checkpoint, count the fraction of checkpoints
    in each state.
    """
    # Get the main chain
    checkpoint = validator.highest_justified_checkpoint

    count_justified = 0
    count_finalized = 0
    count_total = 0
    while checkpoint is not None:
        count_total += 1
        if checkpoint.hash in validator.justified:
            count_justified += 1
        if checkpoint.hash in validator.finalized:
            count_finalized += 1
        checkpoint = validator.get_checkpoint_parent(checkpoint)

    fraction_justified = float(count_justified) / float(count_total)
    fraction_finalized = float(count_finalized) / float(count_total)
    count_forked_justified = len(validator.justified) - count_justified
    fraction_forked_justified = float(count_forked_justified) / float(count_total)
    return fraction_justified, fraction_finalized, fraction_forked_justified

def delay_throughput(network):
    delay = {}
    Edelay = 0.0
    E2delay = 0.0
    for block_hash in network.final_time :
        delay[block_hash] = network.final_time[block_hash] - network.first_proposal_time[block_hash]
    for block_hash in delay :
        Edelay += delay[block_hash]
        E2delay += delay[block_hash]**2
    if len(delay)>0:
        Edelay = Edelay/len(delay)
        E2delay = E2delay/len(delay)
    else:
        Edelay = Edelay
        E2delay = E2delay
    min_time = BLOCK_PROPOSAL_TIME*EPOCH_SIZE*NUM_EPOCH
    max_time = 0
    for block_hash in network.final_time:
        if(network.final_time[block_hash]>=max_time):
            max_time = network.final_time[block_hash]
    for block_hash in network.final_time:
        if(network.first_proposal_time[block_hash]<=min_time):
            min_time = network.first_proposal_time[block_hash]

    if len(network.final_time)>0:
        throughput = (max_time - min_time)/len(network.final_time)
    else:
        throughput = BLOCK_PROPOSAL_TIME*EPOCH_SIZE*NUM_EPOCH  #max_time - min_time 
    return Edelay, E2delay, throughput, len(network.final_time)

def finalization_quartiles(network):
    sumquartiles = [0.0,0.0,0.0,0.0]
    sumsquarequartiles = [0.0,0.0,0.0,0.0]
    len_valid_sums = [0.0,0.0,0.0,0.0]

    Equartiles = [0.0,0.0,0.0,0.0]
    E2quartiles = [0.0,0.0,0.0,0.0]
    stdquartiles = [0.0,0.0,0.0,0.0]

    for blockhash in network.final_quartiles:
        for i in range(4):
            if len(network.final_quartiles[blockhash])>i:
                sumquartiles[i] = sumquartiles[i] + network.final_quartiles[blockhash][i]
                sumsquarequartiles[i] += network.final_quartiles[blockhash][i]**2
                len_valid_sums[i] +=1
    '''
    for i in range(4):            
        if len_valid_sums[i]>0:
            Equartiles[i] = Equartiles[i]/len_valid_sums[i]
            E2quartiles[i] = E2quartiles[i]/len_valid_sums[i]
            stdquartiles[i] = E2quartiles[i] - Equartiles[i]**2
            stdquartiles[i] = np.sqrt(stdquartiles[i])
        else:
            Equartiles[i] = None
            stdquartiles[i] = None
    '''
    print(len_valid_sums)
    return sumquartiles,sumsquarequartiles,len_valid_sums

def main_chain_size(validator):
    """Computes the number of blocks in the main chain."""
    return validator.highest_justified_checkpoint.height + 1


def blocks_under_highest_justified(validator):
    """Computes the height of blocks below the checkpoint of highest height."""
    res = 0
    for bhash, b in validator.processed.items():
        if isinstance(b, Block):
            if b.height <= validator.highest_justified_checkpoint.height:
                res += 1
    return res


def total_height_blocks(validator):
    """Total height of blocks processed by the validator.
    """
    res = 0
    for bhash, b in validator.processed.items():
        if isinstance(b, Block):
            res += 1
    return res


def count_forks(validator):
    """Compute the height of forks of each size.

    Returns a dict {1: 24, 2: 5, 3: 2} for instance.
    Compute forks up until the highest justified checkpoint.
    """
    # Compute a list of the block hashes in the main chain, up to the highest justified checkpoint.
    block = validator.highest_justified_checkpoint
    block_hash = block.hash
    main_blocks = [block_hash]

    # Stop when we reach the genesis block
    while block.height > 0:
        block_hash = block.prevhash
        block = validator.processed[block_hash]
        main_blocks.append(block_hash)

    # Check that we reached the genesis block
    assert block.height == 0
    assert len(main_blocks) == validator.highest_justified_checkpoint.height + 1

    # Now iterate through the blocks with height below highest_justified
    longest_fork = {}
    for block_hash, block in validator.processed.items():
        if isinstance(block, Block):
            if block.height <= validator.highest_justified_checkpoint.height:
                # Get the closest parent of block from the main blockchain
                fork_length = 0
                while block_hash not in main_blocks:
                    fork_length += 1
                    block_hash = block.prevhash
                    block = validator.processed[block_hash]
                assert block_hash in main_blocks
                longest_fork[block_hash] = max(longest_fork.get(block_hash, 0), fork_length)

    count_forks = {}
    for block_hash in main_blocks:
        l = longest_fork[block_hash]
        count_forks[l] = count_forks.get(l, 0) + 1

    assert sum(count_forks.values()) == validator.highest_justified_checkpoint.height + 1
    return count_forks


def print_metrics_latency(latencies, num_tries, validator_set=VALIDATOR_IDS):
    for latency in latencies:
        jfsum = 0.0
        squarejfsum = 0.0
        ffsum = 0.0
        squareffsum = 0.0
        jffsum = 0.0
        squarejffsum = 0.0
        mcsum = 0.0
        squaremcsum = 0.0
        busum = 0.0
        squarebusum = 0.0
        delaysum = 0.0
        squaredelaysum = 0.0
        throughputsum = 0.0
        squarethroughputsum = 0.0
        total_finalized = 0.0
        num_finalized_tries = 0.0
        finalization_achieved = True
        Equartiles = [0.0,0.0,0.0,0.0]
        E2quartiles = [0.0,0.0,0.0,0.0]
        stdquartiles = [0.0,0.0,0.0,0.0]
        len_valid_sums = [0.0,0.0,0.0,0.0]

        #fcsum = {}
        for i in range(num_tries):
            network = Network(exponential_latency(latency))
            validators = [VoteValidator(network, i) for i in validator_set]

            for t in range(BLOCK_PROPOSAL_TIME * EPOCH_SIZE * NUM_EPOCH):
                network.tick()
                # if t % (BLOCK_PROPOSAL_TIME * EPOCH_SIZE) == 0:
                #     filename = os.path.join(LOG_DIR, 'plot_{:03d}.png'.format(t))
                #     plot_node_blockchains(validators, filename)

            for val in validators:
                jf, ff, jff = fraction_justified_and_finalized(val)
                jfsum += jf
                squarejfsum += jf**2
                ffsum += ff
                squareffsum += ff**2
                jffsum += jff
                squarejffsum += jff**2
                mcsum += main_chain_size(val)
                squaremcsum += main_chain_size(val)**2
                busum += blocks_under_highest_justified(val)
                squarebusum += blocks_under_highest_justified(val)**2
                #fc = count_forks(val)
                #for l in fc.keys():
                    #fcsum[l] = fcsum.get(l, 0) + fc[l]

            Edelay,E2delay,throughput,num_finalized = delay_throughput(network)
            delaysum += Edelay*num_finalized
            squaredelaysum += E2delay*num_finalized
            if num_finalized>0:
                throughputsum +=throughput
                num_finalized_tries = num_finalized_tries + 1
                squarethroughputsum += throughput**2
            total_finalized += num_finalized

            tempEquartiles,tempE2quartiles,templen_valid_sums = finalization_quartiles(network)
            for i in range(4):
                if templen_valid_sums[i]>0 :   
                    Equartiles[i] += tempEquartiles[i]
                    E2quartiles[i] += tempE2quartiles[i]
                    len_valid_sums[i] += templen_valid_sums[i]



        if total_finalized > 0 :
            Edelay = delaysum/total_finalized
            E2delay = squaredelaysum/total_finalized
            vardelay = E2delay - Edelay**2
            Ethroughput = throughputsum/num_finalized_tries
            E2throughput = squarethroughputsum/num_finalized_tries
            varthroughput = E2throughput - Ethroughput**2    
        else:
            finalization_achieved = False
            #print('No finalization Achieved')
        
        for i in range(4):            
            if len_valid_sums[i]>0:
                Equartiles[i] = Equartiles[i]/len_valid_sums[i]
                E2quartiles[i] = E2quartiles[i]/len_valid_sums[i]
                stdquartiles[i] = E2quartiles[i] - Equartiles[i]**2
                stdquartiles[i] = np.sqrt(stdquartiles[i])
            else:
                Equartiles[i] = None
                stdquartiles[i] = None


        Ejf = jfsum/len(validators)/num_tries
        E2jf = squarejfsum/len(validators)/num_tries
        Ejff = jffsum/len(validators)/num_tries
        E2jff = squarejffsum/len(validators)/num_tries
        Eff = ffsum/len(validators)/num_tries
        E2ff = squareffsum/len(validators)/num_tries
        Emc = mcsum/len(validators)/num_tries
        E2mc = squaremcsum/len(validators)/num_tries
        Ebu = busum/len(validators)/num_tries
        E2bu = squarebusum/len(validators)/num_tries
        varjf = E2jf - Ejf**2
        varjff = E2jff - Ejff**2
        varff = E2ff - Eff**2
        varmc = E2mc - Emc**2
        varbu = E2bu - Ebu**2


        print('Latency: {}'.format(latency))
        #print('Justified: {}'.format([Ejf,varjf]))
        #print('Finalized: {}'.format([Eff,varff]))
        print('Justified in forks: {}'.format([Ejff,np.sqrt(varjff)]))
        print('Main chain size: {}'.format([Emc,np.sqrt(varmc)]))
        print('Main chain size:{}'.format([Emc/EPOCH_SIZE*NUM_EPOCH, np.sqrt(varmc)/EPOCH_SIZE*NUM_EPOCH ]))
        print('Blocks under main justified: {}'.format([Ebu,varbu]))
        print('finalization_quartiles:{}'.format([Equartiles,stdquartiles]))
        if finalization_achieved :
            print('Delay:{}'.format([Edelay,vardelay]))
            print('Throughput:{}'.format([Ethroughput,varthroughput]))
        else:
            print('No finalization achieved')
        #print('Main chain fraction: {}'.format(
        #    mcsum / (len(validators) * num_tries * (EPOCH_SIZE * NUM_EPOCH + 1))))
        #for l in sorted(fcsum.keys()):
            #if l > 0:
                #frac = float(fcsum[l]) / float(fcsum[0])
                #print('Fraction of forks of size {}: {}'.format(l, frac))
        print('')


if __name__ == '__main__':
    LOG_DIR = 'metrics'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Uncomment to have fractions of disconnected nodes
    # fractions = np.arange(0.0, 0.4, 0.05)
    # fractions = [0.31, 0.32, 0.33]
    fractions = [0.0]
    for fraction_disconnected in fractions:
        num_validators = int((1.0 - fraction_disconnected) * NUM_VALIDATORS)
        validator_set = VALIDATOR_IDS[:num_validators]

        print("Total height of nodes: {}".format(NUM_VALIDATORS))
        print("height of connected of nodes: {}".format(len(validator_set)))

        # Uncomment to have different latencies
        #latencies = [i for i in range(10, 300, 20)] + [500, 750, 1000]
        latencies = [0,50,100,250,500,1000]
        num_tries = 1  # number of samples for each set of parameters

        print_metrics_latency(latencies, num_tries, validator_set)

#!/usr/bin/python3
import os
from termcolor import colored
import numpy as np

from parameters import *
from block import Block
from utils import exponential_latency
from network import Network
from validator import VoteValidator
#from plot_graph import plot_node_blockchains
from collections import Counter
from tqdm import tqdm
from pandas import DataFrame
import random
import networkx as nx
import numpy as np

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
    delay_array = np.array([])
    for block_hash in network.global_finalized_time_absolute :
        delay[block_hash] = network.global_finalized_time_absolute[block_hash]  - network.first_proposal_time[block_hash]
        delay_array = np.append(delay_array,delay[block_hash])
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
    max_time = 1
    max_epoch=0
    for block_hash in network.global_finalized_time_absolute:
        if(network.global_finalized_time_absolute[block_hash]>=max_time):
            max_time = network.global_finalized_time_absolute[block_hash]
    for block_hash in network.global_finalized_time_absolute:
        if(network.first_proposal_time[block_hash]<=min_time):
            min_time = network.first_proposal_time[block_hash]
    for block_hash in network.global_finalized_time_absolute:
        if((network.processed[block_hash].height)/EPOCH_SIZE>=max_epoch):
            max_epoch = (network.processed[block_hash].height)/EPOCH_SIZE

    if len(network.global_finalized_time)>0:
        throughput = max_epoch/(max_time)       #Removed min_time
    else:
        throughput = 0  #max_time - min_time

    return Edelay, E2delay, throughput, len(network.global_finalized_time), delay_array

def timing_chain(network):
    timing = {}
    sum_timing = [0,0,0,0,0,0,0,0]
    timing_count = [0,0,0,0,0,0,0,0]
    for blockhash in network.global_finalized_time:
        timing[blockhash] = [network.first_justification_time.get(blockhash,-1), network.global_justified_time.get(blockhash,-1), \
        network.first_justified_time.get(blockhash,-1), network.last_justified_time.get(blockhash,-1), network.first_finalization_time.get(blockhash,-1), \
        network.global_finalized_time.get(blockhash,-1), network.first_finalized_time.get(blockhash,-1), network.last_finalized_time.get(blockhash,-1)  ]
    for blockhash in timing:
        for i in range(len(timing[blockhash])):
            if timing[blockhash][i] == -1:
                pass
            else:
                sum_timing[i] = sum_timing[i] + timing[blockhash][i]
                timing_count[i] = timing_count[i] + 1

    return np.array(sum_timing), np.array(timing_count)

'''

def depth_calculator(validator):
    depth_finalized = 0
    num_depth_finalized =0
    main_chain_size = validator.main_chain_size
    max_epoch=0
    for blockhash in validator.finalized:
        if validator.processed[blockhash].height > max_epoch:
            max_epoch = validator.processed[blockhash].height
        num_depth_finalized = 1
    depth_finalized = (main_chain_size - max_epoch)/EPOCH_SIZE
    return depth_finalized, num_depth_finalized

'''


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


def print_metrics_latency (G, num_tries, latencies,
                          wait_fractions, vote_as_block, immediate_vote, validator_set=VALIDATOR_IDS):
    # metrics for analysis
    tp      = []
    delay   = []
    depth   = []
    mcf     = []

    for latency in latencies:

        #fcsum = {}
        new_delay_list      = []
        old_delay_list      = []
        tp_list             = []
        depth_list          = []
        wait_fraction_list  = []
        MCF_list            = []
        mcf_list_percent    = []

        for wf in wait_fractions:

            print('wait_fraction: {}'.format(wf))

            sum_type_1_vote         = 0
            sum_type_2_vote         = 0
            mining_ids              = [i for i in range(NUM_VALIDATORS)]
            jfsum                   = 0.0
            squarejfsum             = 0.0
            ffsum                   = 0.0
            squareffsum             = 0.0
            jffsum                  = 0.0
            squarejffsum            = 0.0
            mcsum                   = 0.0
            squaremcsum             = 0.0
            busum                   = 0.0
            squarebusum             = 0.0
            delaysum                = 0.0
            squaredelaysum          = 0.0
            throughputsum           = 0.0
            squarethroughputsum     = 0.0
            total_finalized         = 0.0
            num_finalized_tries     = 0.0
            finalization_achieved   = True
            Equartiles              = [0.0,0.0,0.0,0.0]
            E2quartiles             = [0.0,0.0,0.0,0.0]
            stdquartiles            = [0.0,0.0,0.0,0.0]
            len_valid_sums          = [0.0,0.0,0.0,0.0]
            Etiming                 = np.zeros((8))
            timing_count            = np.zeros((8))
            sml_stats               = {}
            depth_finalized         = 0
            num_depth_finalized     = 0
            delay_array             = np.array([])   

            for i in range(num_tries):
                network = Network(G.adj, latency)
                validators = [VoteValidator(network,latency,wf, i,vote_as_block, immediate_vote) for i in validator_set]

                for t in tqdm(range(BLOCK_PROPOSAL_TIME * EPOCH_SIZE * NUM_EPOCH)):
                    network.tick(sml_stats)
                    # The code below does random shuffling of mining order
                    if t%BLOCK_PROPOSAL_TIME*NUM_VALIDATORS ==0:
                        random.shuffle(mining_ids)
                        for val in validators:
                            val.mining_id = mining_ids[val.id]

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
                    temp_depth_finalized = val.depth_finalized
                    temp_num_depth_finalized = val.num_depth_finalized
                    num_depth_finalized += temp_num_depth_finalized
                    depth_finalized += temp_depth_finalized
                    sum_type_1_vote += val.type_1_vote
                    sum_type_2_vote += val.type_2_vote

                    #fc = count_forks(val)
                    #for l in fc.keys():
                        #fcsum[l] = fcsum.get(l, 0) + fc[l]

                Edelay,E2delay,throughput,num_finalized, temp_delay_array = delay_throughput(network)
                delay_array = np.append(delay_array, temp_delay_array)
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

                temp_timing,temp_timing_count = timing_chain(network)
                Etiming += temp_timing
                timing_count += temp_timing_count

            sd_delay = 0
            if total_finalized > 0 :
                Edelay = delaysum/total_finalized
                E2delay = squaredelaysum/total_finalized
                vardelay = E2delay - Edelay**2
                sd_delay = np.sqrt(vardelay)
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


            Ejf = jfsum/len(validators)
            E2jf = squarejfsum/len(validators)
            Ejff = jffsum/len(validators)
            E2jff = squarejffsum/len(validators)
            Eff = ffsum/len(validators)
            E2ff = squareffsum/len(validators)
            Emc = mcsum/len(validators)
            E2mc = squaremcsum/len(validators)
            Ebu = busum/len(validators)
            E2bu = squarebusum/len(validators)
            varjf = E2jf - Ejf**2
            varjff = E2jff - Ejff**2
            varff = E2ff - Eff**2
            varmc = E2mc - Emc**2
            varbu = E2bu - Ebu**2

            depth_finalized = depth_finalized/num_depth_finalized

            Etiming = Etiming/timing_count

            tp += [Ethroughput]
            delay += [Edelay/(Emc/(EPOCH_SIZE*NUM_EPOCH + 1))]
            depth += [depth_finalized]
            mcf += [Emc/(EPOCH_SIZE*NUM_EPOCH + 1)]

            delay_array = delay_array/(Emc/(EPOCH_SIZE*NUM_EPOCH + 1))

            delay_quartiles = [np.percentile(delay_array,25),
                              np.percentile(delay_array,50),
                              np.percentile(delay_array,75),
                              np.percentile(delay_array,100)]

            print('=== Statistics ===')
            print('Latency: {}'
                    .format(latency))
            print('wait fraction: {}'
                    .format(wf))
            print('Justified in forks: {}'
                    .format([Ejff,np.sqrt(varjff)]))
            print('Main chain size (root included): {}'
                    .format([Emc,np.sqrt(varmc)]))
            # include ROOT
            print('Main chain fraction:{}'
                    .format([Emc/(EPOCH_SIZE*NUM_EPOCH + 1),
                            np.sqrt(varmc)/EPOCH_SIZE/NUM_EPOCH ]))
            # include ROOT
            print(colored('Main chain fraction (%):{}'
                    .format(100*Emc/(EPOCH_SIZE*NUM_EPOCH + 1)), "green"))
            print('type_1_vote: {}'.format(sum_type_1_vote))
            print('type_2_vote: {}'.format(sum_type_2_vote))
            
            if finalization_achieved :
                print('---new, incld. dead blocks---')
                print(colored('Delay:{}'
                        .format(np.mean(delay_array)), "green"))
                print('Delay_quartiles:{}'
                        .format(delay_quartiles))
                print('sd_delay:{}'.format(sd_delay/(Emc/(EPOCH_SIZE*NUM_EPOCH + 1))))
                print('Throughput:{}'
                        .format(Ethroughput))
                print('depth:{}'
                        .format(depth_finalized))
            else:
                print('No finalization achieved')
            print('supermajority link stats: {}'
                    .format(dict(Counter(sml_stats.values()))))
            print('proposer history {}'
                    .format(network.proposer_history))
            print('=== END === ')

            wait_fraction_list.append(wf)
            new_delay_list.append(Edelay/(Emc/(EPOCH_SIZE*NUM_EPOCH + 1)))
            old_delay_list.append(Edelay)
            tp_list.append(Ethroughput)
            depth_list.append(depth_finalized)
            MCF_list.append(Emc/(EPOCH_SIZE*NUM_EPOCH + 1))
            mcf_list_percent.append(100*Emc/(EPOCH_SIZE*NUM_EPOCH + 1))

        print('wait_fraction_list: {}'.format(wait_fraction_list))
        print(colored('delay_list: {}'.format(new_delay_list),"green"))
        print(colored('mcf_list_percent: {}'.format(mcf_list_percent),"green"))
        print('depth_list: {}'.format(depth_list))

        return (tp, delay, depth, mcf)


if __name__ == '__main__':
    # LOG_DIR = 'metrics'
    # if not os.path.exists(LOG_DIR):
        # os.makedirs(LOG_DIR)

    # Uncomment to have fractions of disconnected nodes
    # fractions = np.arange(0.0, 0.4, 0.05)
    # fractions = [0.31, 0.32, 0.33]
    fractions = [0.0]

    print('``````````````````')
    print("""running test
            PROTOCOL: vote-on-majority-after-delay
            NUM_EPOCH: {}
            SUPER_MAJORITY: {}
            NUM_VALIDATORS: {}
            D_REGULAR: {}
            Vote_as_block: {}
            Immediate_vote: {}
            PROPOSAL_LOTT: {}""".
            format(NUM_EPOCH,
                   SUPER_MAJORITY,
                   NUM_VALIDATORS,
                   D_REGULAR,
                   VOTE_AS_BLOCK,
                   IMMEDIATE_VOTE,
                   PROPOSAL_LOTT))
    print('``````````````````')

    for fraction_disconnected in fractions:
        num_validators = int((1.0 - fraction_disconnected) * NUM_VALIDATORS)
        validator_set = VALIDATOR_IDS[:num_validators]

        print("height of connected of nodes: {}".format(len(validator_set)))

        num_tries = 1
        #latencies = [i for i in range(10, 300, 20)] + [500, 750, 1000]
        latencies = [250]
        wait_fractions =  [0.0, 0.1, 0.2, 0.5, 1.0]
        

        # Graph
        # G = nx.complete_graph(num_validators)                # fully connected
        G = nx.random_regular_graph(D_REGULAR,num_validators)  # d-regular

        tp, delay, depth, mcf = print_metrics_latency(G, num_tries,latencies, \
                                                      wait_fractions,
                                                        VOTE_AS_BLOCK,
                                                        IMMEDIATE_VOTE,
                                                        validator_set)

        # save data to test.xlsx
        df = DataFrame({"latency":latencies,
                        "throughput":tp,
                        "mcf": mcf,
                        "delay":delay,
                        "depth":depth
                        },
                        index = wait_fractions).T

        df.to_excel('test.xlsx',sheet_name='sheet1',index=False)

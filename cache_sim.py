import argparse
import numpy as np
import pandas as pd
import time

NUM_KB = 2
CACHE_SIZE = NUM_KB*1024 #bytes
BLOCK_SIZE = 64 #bytes
ASSOCIATIVITY = 4
NUM_SETS = int(CACHE_SIZE/BLOCK_SIZE/ASSOCIATIVITY)

cache = np.negative(np.ones((NUM_SETS,ASSOCIATIVITY)).astype('float'))
times = np.negative(np.ones((NUM_SETS,ASSOCIATIVITY)).astype('float'))

#Measurement variables
HIT_TIME = 1
MISS_TIME = 250

def update_times(set, way):
    [[(t + 1) if (t != -1) else t for t in times[i]] for i in range(len(times))]
    times[set][way] = 0

def find_set(address):
    real_set = int(address/BLOCK_SIZE)
    set = real_set % NUM_SETS
    return set

def add_to_cache(address):
    set = find_set(address)
    address = int(address/BLOCK_SIZE)*BLOCK_SIZE
    loc = np.argwhere(times[set] == -1).flatten()
    if (len(loc) == 0): # need to replace LRU
        #print("evict")
        LRU = max(times[set])
        LRU_loc = np.argwhere(times[set] == LRU).flatten()[0]
        cache[set][LRU_loc] = address
        update_times(set, LRU_loc)
    else: # cache is not full
        add_loc = loc[0]
        cache[set][add_loc] = address
        update_times(set, add_loc)

def check_cache(address, hits, misses, latency):
    set = find_set(address)
    address = int(address/BLOCK_SIZE)*BLOCK_SIZE
    way = np.argwhere(cache[set] == address).flatten()

    if (len(way) == 0): #miss
        result = 0
        misses = misses + 1
        latency = latency + 1 + MISS_TIME

        add_to_cache(address)
    else: #hit
        result = 1
        hits = hits + 1
        latency = latency + 1 + HIT_TIME

        way = way[0]
        update_times(set, way)

    return result, hits, misses, latency

def search_cache(address):
    set = find_set(address)
    address = int(address/BLOCK_SIZE)*BLOCK_SIZE
    way = np.argwhere(cache[set] == address).flatten()
    if (len(way) == 0): #miss
        return 0
    else: # hit
        return 1

def print_cache():
    cache_string = ""
    for s in range(NUM_SETS):
        for a in range(ASSOCIATIVITY):
            cache_string = cache_string + str(cache[s][a]) + " "
        cache_string = cache_string + "\n"
    print(cache_string)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to data file")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    args = parser.parse_args()
    return args

def cachesim(data, hits, misses, latency, filename):
    for i in range(len(data)):
        result, hits, misses, latency = check_cache(data[i], hits, misses, latency)
        array_to_file.append(str(data[i]) + "," + str(result) + "," + str(hits) + "," + str(misses) + "," + str(latency))
        #print_cache()

    with open(filename, 'w+') as file:
        file.write("Address,Result,Hits,Misses,Latency\n")
        for item in array_to_file:
            file.write("%s\n" % item)

    print("Misses = " + str(misses))
    print("Latency = " + str(latency))

def predict_cachesim(data, hits, misses, latency, filename):
    for i in range(len(data)):
        address = data[i]
        in_cache = search_cache(address)
        print(in_cache)
        if (in_cache):
            result, hits, misses, latency = add_to_cache(address, hits, misses, latency)
        else:
            result = 0
            misses = misses + 1
            latency = latency + 251
        array_to_file.append(str(address) + "," + str(result) + "," + str(hits) + "," + str(misses) + "," + str(latency))
        #print_cache()

    with open(filename, 'w+') as file:
        file.write("Address,Result,Hits,Misses,Latency\n")
        for item in array_to_file:
            file.write("%s\n" % item)

    print("Misses = " + str(misses))
    print("Latency = " + str(latency))

if __name__ == '__main__':

    # Start timing
    start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    # Read data
    csv_file = args.file
    df = pd.read_csv(csv_file, dtype={'address':'float'})
    data = df['address'].to_numpy().flatten()

    array_to_file = list()

    hits = misses = latency = 0

    # Simulate
    cachesim(data, hits, misses, latency, args.output)

    #End timing
    end_time = time.time()
    elapsed = round(end_time - start_time)
    print ("Elapsed Time = " + str(elapsed))

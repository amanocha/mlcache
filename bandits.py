"""
Aninda Manocha
May 8, 2019

This code was modeled after the example illustrated in https://github.com/david-cortes/contextualbandits/blob/master/example/online_contextual_bandits.ipynb.
"""

from contextualbandits.online import AdaptiveGreedy, BootstrappedUCB, BootstrappedTS, SoftmaxExplorer
from sklearn.linear_model import LogisticRegression
from cache_sim import *

import argparse
from copy import deepcopy
from math import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

prefix = "output/moreno/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/moreno_divide.csv", help="Path to data file")
    parser.add_argument("--model_num", type=int, default=0, help="Model to run")
    parser.add_argument("--distance", type=int, default=0, help="Prefetch distance")
    args = parser.parse_args()
    return args

def simulate_rounds(model, predictions, rewards, actions, X, y, start, distance, results, hits, misses, latency):
    # Predict
    address = X[start][0]
    curr_actions = model.predict(X[start+distance:start+distance+1, :]).astype('uint8')
    prediction = curr_actions[0]
    predictions.append(prediction)
    past_pred = predictions[len(predictions)-1-distance]

    if (not past_pred):
        if (search_cache(address)): #misprediction
            result, hits, misses, latency = check_cache(address, hits, misses, latency)
        else:
            add_to_cache(address)
            result = 1
            hits = hits + 1
            latency = latency + 251 - distance
    else:
        # Get actual result
        result, hits, misses, latency = check_cache(address, hits, misses, latency)

    results.append(result)

    # Calculate rewards and store
    #reward = # sum((y[np.arange(start, end),1] == curr_actions).astype('uint8'))
    if (past_pred != -1):
        correct = (past_pred == result).astype('uint8')
        rewards = np.append(rewards, correct)
    #else:
        #rewards = np.append(rewards, 0)
    #reward = (result == curr_actions[0]).astype('uint8')
    #rewards.append(reward)

    # Store data from iteration for fitting
    new_actions = np.append(actions, curr_actions)
    #new_results = (y[np.arange(end),1] == new_actions).astype('uint8')
    new_rewards = (results == new_actions).astype('uint8')

    # Fit model again
    model.fit(X[:start+1, :], new_actions, new_rewards)

    return new_actions, results, hits, misses, latency, predictions, rewards

def get_mean_reward(rewards, block_size):
    means=list()
    length = len(rewards)
    for i in range(length):
        means.append(sum(rewards[:i+1]) * 1.0 / ((i+1)*block_size))
        #means.append(sum(rewards[i])*1.0/block_size)
    return means

def plot_results(rewards,block_size):
    colors=plt.cm.tab20(np.linspace(0, 1, len(rewards)))
    '''
    labels = ["Adaptive Greedy (decaying threshold)",
              "Adaptive Greedy (p0=30%, decaying percentile)",
              "Adaptive Active Greedy",
              "Bootstrapped Thompson Sampling",
              "Bootstrapped Upper-Confidence Bound (C.I.=80%)",
              "Softmax Explorer"
              ]
    '''

    labels = ["model"]

    length = len(rewards)
    for i in range(length):
        reward = rewards[i]
        plt.plot(get_mean_reward(reward,block_size), label=labels[i],color=colors[i])

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Mean Reward')
    plt.show()

if __name__ == '__main__':

    # Start timing
    start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    # Read processed data
    csv_file = args.file
    df = pd.read_csv(csv_file, low_memory=False)
    X = df.drop(['latency','hit'], axis=1).to_numpy()
    y = df[['latency','hit']].to_numpy()

    num_samples, num_features = X.shape[0], X.shape[1]

    # Contextual bandits setup
    nchoices = 2 #hit or miss
    base_algorithm = LogisticRegression(random_state=0, solver='lbfgs')
    block_size = 1
    max = 10
    alpha = max/2
    beta = max/2
    beta_prior = ((alpha, beta), max/2)

    # Models

    model_opts = [AdaptiveGreedy(deepcopy(base_algorithm), nchoices=nchoices, decay_type='threshold'),
                  AdaptiveGreedy(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, decay_type='percentile', decay=0.9997),
                  AdaptiveGreedy(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, active_choice='weighted', decay_type='percentile', decay=0.9997),
                  BootstrappedTS(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior),
                  BootstrappedUCB(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior),
                  SoftmaxExplorer(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior)
                  ]

    model_names_opts = ["adaptive_greedy_thres", "adaptive_greedy_perc", "adaptive_active_greedy", "bootstrapped_ts", "bootstrapped_ucb", "softmax"]

    '''
    models = [AdaptiveGreedy(deepcopy(base_algorithm), nchoices=nchoices, decay_type='threshold')]
    models = [AdaptiveGreedy(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, decay_type='percentile', decay=0.9997)]
    models = [AdaptiveGreedy(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, active_choice='weighted', decay_type='percentile', decay=0.9997)]
    #models = [BootstrappedTS(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior)]

    model_names = ["adaptive_greedy_thres"]
    model_names = ["adaptive_greedy_perc"]
    model_names = ["adaptive_active_greedy"]
    #model_names = ["bootstrapped_ts"]
    '''

    models = [model_opts[args.model_num]]
    model_names = [model_names_opts[args.model_num]]

    # Initial fitting
    first_batch = X[:block_size, :]
    address = X[0][0]
    action_chosen = np.random.randint(nchoices, size=block_size)

    result, hit, miss, latency = check_cache(address, 0, 0, 0)
    rewards = np.array([(result == action_chosen[0]).astype('uint8')])
    rewards_list = [rewards for m in range(len(models))]
    #rewards = (y[:block_size,1] == action_chosen).astype('uint8')

    for model in models:
        np.random.seed(0)
        model.fit(X=first_batch, a=action_chosen, r=rewards)

    actions_list = [action_chosen.copy() for i in range(len(models))]

    predictions = [[action_chosen[0]] for m in range(len(models))]
    results = [[result] for m in range(len(models))]
    hits = [hit for i in range(len(models))]
    misses = [miss for i in range(len(models))]
    latencies = [latency for i in range(len(models))]

    distance = args.distance
    for i in range(1,distance+1):
        for model in range(len(models)):
            predictions[model].append(-1)
            rewards_list[model] = np.append(rewards_list[model], 0)

    # Run simulation
    length = 3001 #int(np.floor(num_samples / block_size))

    for m in range(len(models)):
        name = model_names[m]
        model_file = open(prefix+name+str(distance)+'.csv', 'w+')
        model_file.write("Iteration, Result, Prediction, Reward, Cumulative Mean, Latency\n")
        model_file.write("0," + str(result) + "," + str(predictions[m][0]) + "," + str(rewards_list[m][0]) + "," + str(get_mean_reward(rewards_list[m],block_size)[0]) + "," + str(latencies[m]) + "\n")
        model_file.close()

    print "Finished initialization!"

    for i in range(block_size, length-distance):
        if (i % 100 == 0):
            print i
        #start = (i + 1) * block_size
        #end = np.min([start+block_size, num_samples])
        for model in range(len(models)):
            name = model_names[model]
            '''
            actions_m = actions_list[model]
            results_m = results[model]
            hits_m = hits[model]
            misses_m = misses[model]
            latencies_m = latencies[model]
            predictions_m = predictions[model]
            rewards_m = rewards_list[model]
            '''
            model_file = open(prefix+name+str(distance)+'.csv', 'a+')

            actions_list[model], results[model], hits[model], misses[model], latencies[model], predictions[model], rewards_list[model] = simulate_rounds(models[model], predictions[model], rewards_list[model], actions_list[model], X, y, i, distance, results[model], hits[model], misses[model], latencies[model])
            filestring = str(i) + "," + str(results[model][i]) + "," + str(predictions[model][i]) + "," + str(rewards_list[model][i]) + "," + str(get_mean_reward(rewards_list[model],block_size)[i]) + "," + str(latencies[model]) + "\n"
            model_file.write(filestring)
            model_file.close()

    for i in range(length-distance, length):
        for m in range(len(models)):
            address = X[i][0]

            if (not predictions[m][i]):
                if (search_cache(address)): #misprediction
                    result, hits[m], misses[m], latencies[m] = check_cache(address, hits[m], misses[m], latencies[m])
                else:
                    add_to_cache(address)
                    result = 1
                    hits[m] = hits[m] + 1
                    latencies[m] = latencies[m] + 251 - distance
            else:
                # Get actual result
                result, hits[m], misses[m], latencies[m] = check_cache(address, hits[m], misses[m], latencies[m])

            results[m].append(result)
            actual = results[m][len(results[m])-1-distance]
            correct = (predictions[m][i] == actual).astype('uint8')
            rewards_list[m] = np.append(rewards_list[m],correct)

            name = model_names[m]
            model_file = open(prefix+name+str(distance)+'.csv', 'a+')
            filestring = str(i) + "," + str(results[m][i]) + "," + str(predictions[m][i]) + "," + str(rewards_list[m][i]) + "," + str(get_mean_reward(rewards_list[m], block_size)[i]) + "," + str(latencies[m]) + "\n"
            model_file.write(filestring)
            model_file.close()

    # Plot
    #plot_results(rewards_list,block_size)

    #End timing
    end_time = time.time()
    elapsed = round(end_time - start_time)
    print ("Elapsed Time = " + str(elapsed))

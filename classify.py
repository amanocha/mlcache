from core import *
from cache_sim_big import *

import argparse
from math import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

prefix = "output/moreno_big/classify/"

model_opts = [decision_tree(), random_forest(), knn(), nb()]
model_names_opts = ["decision_tree", "random_forest", "knn", "nb"]
labels = ["Decision Tree", "Random Forest", "K-Nearest Neighbors", "Multinomial Naive Bayes"]

labels = ["model"]

num_iterations = 5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/moreno_divide.csv", help="Path to data file")
    parser.add_argument("--model_num", type=int, default=0, help="Model to run")
    parser.add_argument("--distance", type=int, default=0, help="Prefetch distance")
    args = parser.parse_args()
    return args

def get_mean_reward(rewards):
    means=list()
    length = len(rewards)
    for i in range(length):
        means.append(sum(rewards[:i+1]) * 1.0 / (i+1))
    return means

def plot_results(rewards):
    colors=plt.cm.tab20(np.linspace(0, 1, len(rewards)))

    length = len(rewards)
    for i in range(length):
        reward = rewards[i]
        plt.plot(get_mean_reward(reward), label=labels[i],color=colors[i])

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Mean Reward')
    plt.show()

def initialize(X, num_iterations, models, model_names, distance):
    hits = [0 for i in range(len(models))]
    misses = [0 for i in range(len(models))]
    latencies = [0 for i in range(len(models))]
    rewards = [list() for i in range(len(models))]
    predictions = [list() for i in range(len(models))]
    y = [np.array([]) for i in range(len(models))]

    for i in range(distance+num_iterations):
        for m in range(len(models)):
            predictions[m].append(-1)
            rewards[m].append(0)

    for i in range(num_iterations):
        for m in range(len(models)):
            address = X[i][0]
            result, hits[m], misses[m], latencies[m] = check_cache(address, hits[m], misses[m], latencies[m])
            y[m] = np.append(y[m],result)

            name = model_names[m]
            model_file = open(prefix+name+str(distance)+'.csv', 'a+')
            filestring = str(i) + "," + str(y[m][i]) + "," + str(predictions[m][i]) + "," + str(rewards[m][i]) + "," + str(get_mean_reward(rewards[m])[i]) + "," + str(latencies[m]) + "\n"
            model_file.write(filestring)
            model_file.close()

    return y, hits, misses, latencies, rewards, predictions

def finish(X, y, hits, misses, latency, rewards, predictions, models, model_names, distance):
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

            y[m] = np.append(y[m],result)
            actual = y[m][len(y[m])-1-distance]
            correct = (predictions[m][i] == actual).astype('uint8')
            rewards[m].append(correct)

            name = model_names[m]
            model_file = open(prefix+name+str(distance)+'.csv', 'a+')
            filestring = str(i) + "," + str(y[m][i]) + "," + str(predictions[m][i]) + "," + str(rewards[m][i]) + "," + str(get_mean_reward(rewards[m])[i]) + "," + str(latencies[m]) + "\n"
            model_file.write(filestring)
            model_file.close()

    #return y, hits, misses, latencies, rewards, predictions

def adj_cache(prediction, address, hits, misses, latency):
    if (prediction): #predict hit
        result, hits, misses, latency = check_cache(address, hits, misses, latency)
    else: #predict miss
        result = search_cache(address)
        if (result): # actual hit
            hits = hits + 1
            latency = latency + 2
        else: # actual miss
            misses = misses + 1
            latency = latency + 251
    return result, hits, misses, latency

def predict(model, X, y, new, hits, misses, latency, rewards, predictions, distance):
    address = new[0][0] #access to predict
    prediction = model.predict(new)[0]
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

    X = np.vstack((X,new))
    y = np.append(y,result)
    model.fit(X, y)

    if (past_pred != -1):
        correct = (past_pred == result).astype('uint8')
        rewards.append(correct)

    return X, y, hits, misses, latency, rewards, predictions

if __name__ == '__main__':

    # Start timing
    start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    # Read processed data
    csv_file = args.file
    df = pd.read_csv(csv_file, low_memory=False)
    data = df.drop(['latency','hit'], axis=1).to_numpy()

    models = [model_opts[args.model_num]]
    model_names = [model_names_opts[args.model_num]]
    distance = args.distance

    for m in range(len(models)):
        name = model_names[m]
        model_file = open(prefix+name+str(distance)+'.csv', 'w+')
        model_file.write("Iteration, Result, Prediction, Reward, Cumulative Mean, Latency\n")
        model_file.close()

    # Initialization
    X = data[0:num_iterations]
    y, hits, misses, latencies, rewards, predictions = initialize(X, num_iterations, models, model_names, distance)
    print("Finished initialization!")

    # Fit classifiers
    for m in range(len(models)):
        model = models[m]
        model.fit(X, y[m])
        predictions[m]

    length = 3001 #len(data)
    for i in range(num_iterations, length-distance):
        for m in range(len(models)):
            name = model_names[m]
            model_file = open(prefix+name+str(distance)+'.csv', 'a+')
            X, y[m], hits[m], misses[m], latencies[m], rewards[m], predictions[m] = predict(models[m], X, y[m], data[i+distance].reshape(1,-1), hits[m], misses[m], latencies[m], rewards[m], predictions[m], distance)
            filestring = str(i) + "," + str(y[m][i]) + "," + str(predictions[m][i]) + "," + str(rewards[m][i]) + "," + str(get_mean_reward(rewards[m])[i]) + "," + str(latencies[m]) + "\n"
            model_file.write(filestring)
            model_file.close()

    finish(data, y, hits, misses, latencies, rewards, predictions, models, model_names, distance)

    # Plot results
    #plot_results(rewards)

    #End timing
    end_time = time.time()
    elapsed = round(end_time - start_time)
    print ("Elapsed Time = " + str(elapsed))

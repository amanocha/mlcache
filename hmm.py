from hmmlearn import hmm

import argparse
from math import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/divide_address_test.csv", help="Path to data file")
    args = parser.parse_args()
    return args

def split_df(df):
    pivot = int(df.shape[0]*0.8)
    train = df.iloc[0:pivot, :]
    test = df.iloc[pivot:, :]
    return train, test

if __name__ == '__main__':

    # Start timing
    start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    # Read processed data
    csv_file = args.file
    df = pd.read_csv(csv_file, low_memory=False)
    X = df.drop(['latency','hit'], axis=1).to_numpy()
    y = df[['latency','hit']]

    X = X.astype('float64')
    gaussian = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)
    gaussian.fit(X)
    seq = gaussian.predict(X)
    print(seq)

    #End timing
    end_time = time.time()
    elapsed = round(end_time - start_time)
    print ("Elapsed Time = " + str(elapsed))

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

from core import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/moreno.csv", help="Path to data file")
    parser.add_argument("--num_clusters", type=int, default=(2**10), help="Number of clusters to partition the address space")
    parser.add_argument("--threshold", type=int, default=10, help="Minimum number of times an address needs to accessed to be added to the training space")
    parser.add_argument("--mode", type=str, required=True, help="Method of organizing address space (cluster, divide, threshold)")
    args = parser.parse_args()
    return args

def split_df(df):
    pivot = int(df.shape[0]*0.8)
    train = df.iloc[0:pivot, :]
    test = df.iloc[pivot:, :]
    return train, test

def adjust_addresses(df):
    minimum = df['address'].min()
    blocksize = 64
    df['address_norm'] = df.apply(lambda row: int((row['address']-minimum)/blocksize)*blocksize, axis=1)
    new_df = df.drop(['issue_cycle', 'return_cycle'], axis=1)
    return new_df

def divide_addresses(df, num_clusters):
    maximum = df['address_norm'].max()
    divisor = int(maximum/num_clusters)
    df['address_cluster'] = df.apply(lambda row: int(row['address_norm']/divisor), axis=1)
    return df

def encode_addresses(df, col):
    new_df = pd.get_dummies(df, columns = col)
    print(new_df.to_numpy().shape)
    return new_df

def cluster_addresses(df, num_clusters):
    features = df.to_numpy()
    kmeans = kmeans_cluster(n_clusters)
    kmeans.fit()
    new_features = kmeans.transform(features)
    labels = kmeans.labels_
    print(labels)
    return new_features, labels

def threshold_address(df, threshold):
    counts = df['address_norm'].value_counts()
    new_df = df[df.groupby('address_norm')['address_norm'].transform('count').ge(threshold)]
    indices = np.arange(len(counts))
    addresses = [counts.index[i] for i in indices if counts[counts.index[i]] >= threshold]
    return new_df, addresses

# Find number of components for PCA for dimension reduction
def pca_n_components(features):
    pca, features_rescaled = PCA_decomposition(features, len(features[0]))
    explained_variances = np.cumsum(pca.explained_variance_ratio_)

    n_components = np.argmax(explained_variances >= 0.95)
    print(n_components, explained_variances[n_components])

    # Identify how many components should be used
    plt.figure()
    plt.plot(explained_variances)
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Percentage of Variance Explained By PCA Components')
    plt.show()

    return pca, features_rescaled, n_components

def perform_PCA(features, n_components, feature_names, filename):
    pca, features_rescaled = PCA_decomposition(features, n_components)
    new_features = pca.transform(features_rescaled)
    components = pca.components_
    score = pca.score(features)

    pc_cols = [("PC " + str(i)) for i in np.arange(n_components)]
    feat_df = pd.DataFrame(new_features, columns=pc_cols)
    feat_df.to_csv(filename+"_pca_feat.csv", index=False)
    new_df = pd.DataFrame(components, columns=feature_names)
    new_df.to_csv(filename+"_pca_comp.csv", index=False)

    return feat_df, new_df

if __name__ == '__main__':

    # Start timing
    start = time.time()

    # Parse command line arguments
    args = parse_args()

    data_dir = "./data/"
    out_dir = "./output/"

    csv_file = args.file

    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    df = pd.read_csv(csv_file, low_memory=False)
    df = adjust_addresses(df)

    if (args.mode == "cluster"):
        new_features, labels = cluster_addresses(train_df, args.num_clusters)
    elif (args.mode == "divide"):
        df = divide_addresses(df, args.num_clusters)
        df = encode_addresses(df, ['address_cluster'])
        filename = os.path.splitext(args.file)[0]
        df.to_csv(filename + "_divide.csv", index=False)
        df = df.drop(['latency', 'hit'], axis=1)
        names = list(df)

        X = df.to_numpy().astype('float64')
        #pca, features_rescaled, n_components = pca_n_components(X)
        n_components = 400
        feat, comp = perform_PCA(X, n_components, names, filename)
    elif (args.mode == "threshold"):
        #train_df, test_df = split_df(df)
        df, addresses = threshold_address(df, args.threshold)
        df = encode_addresses(df, ['address_norm'])
    else:
        print("Invalid preprocessing mode.")

    end = time.time()
    elapsed = end - start
    print("Elapsed Time = " + str(elapsed))

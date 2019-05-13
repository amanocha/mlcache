import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import *

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis, LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler

## Models for unsupervised learning

# KMeans Clustering
def kmeans_cluster(n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=0, max_iter=500)

# PCA
def PCA_decomposition(features, n_components=None):
    scaler = MinMaxScaler(feature_range=[0, 1])
    features_rescaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    pca.fit(features_rescaled)
    return pca, features_rescaled

## Models for supervised learning

# Decision Trees
def decision_tree():
    return tree.DecisionTreeClassifier()

def random_forest():
    return RandomForestClassifier(n_estimators=20)

def logistic_regression():
    return LogisticRegression(random_state=0, solver='lbfgs')

def knn():
    return KNeighborsClassifier(n_neighbors = 3)

def svm():
    return SVC(kernel='linear')

def perceptron():
    return Perceptron(tol=1e-3, random_state=0, early_stopping=True, validation_fraction=0.2, max_iter=100)

def nb():
    return MultinomialNB()

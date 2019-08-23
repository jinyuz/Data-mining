import numpy as np

def norm(x):
    """
    >>> Function you should not touch
    """
    max_val = np.max(x, axis=0)
    x = x/max_val
    return x

def rand_center(data,k):
    """
    >>> Function you need to write
    >>> Select "k" random points from "data" as the initial centroids.
    """
    pass

def converged(centroids1, centroids2):
    """
    >>> Function you need to write
    >>> check whether centroids1==centroids
    >>> add proper code to handle infinite loop if it never converges
    """
    pass

def update_centroids(data, centroids, k=3):
    """
    >>> Function you need to write
    >>> Assign each data point to its nearest centroid based on the Euclidean distance
    >>> Update the cluster centroid to the mean of all the points assigned to that cluster
    """
    pass

def kmeans(data,k=3):
    """
    >>> Function you should not touch
    """
    # step 1:
    centroids = rand_center(data,k)
    converge = False
    while not converge:
        old_centroids = np.copy(centroids)
        # step 2 & 3
        centroids, label = update_centroids(data, old_centroids)
        # step 4
        converge = converged(old_centroids, centroids)
    print(">>> final centroids")
    print(centroids)
    return centroids, label

def evaluation(predict, ground_truth):
    """
    >>> use F1 and NMI in scikit-learn for evaluation
    """
    pass

def gini(predict, ground_truth):
    """
    >>> use the ground truth to do majority vote to assign a flower type for each cluster
    >>> accordingly calculate the probability of missclassifiction and correct classification
    >>> finally, calculate gini using the calculated probabilities
    """
    pass

def SSE(centroids, data):
    """
    >>> Calculate the sum of squared errors for each cluster
    """
    pass

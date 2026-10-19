import random
import numpy as np

class Kmeans:
    def __init__(self,n_clusters=2,max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        
    def fit_predict(self,X):
        
        random_inx = random.sample(range(0,X.shape[0]),self.n_clusters)
        self.centroids = X[random_inx]
        
        for i in range(self.max_iter):
            cluster_grp = self.assign_clusters(X)
            old_centroid = self.centroids.copy()
            
            self.centroids = self.move_centraoid(X,cluster_grp)
            
            if np.array_equal(old_centroid ,self.centroids):
                break
            
        return cluster_grp
        
    def assign_clusters(self,X):
        cluster_grp = []
        distances = []
        
        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distances = min(distances)
            index_pos = distances.index(min_distances)
            cluster_grp.append(index_pos)
            distances.clear()
        return np.array(cluster_grp)
    
    def move_centraoid(self,X,cluster_grp):
        new_centraoids = []
        cluster_type = np.unique(cluster_grp)
        
        for type in cluster_type:
            new_centraoids.append(X[cluster_grp == type].mean(axis=0))
            
        return new_centraoids
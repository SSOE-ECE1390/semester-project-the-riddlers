import numpy as np
#import kmedoids
from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import euclidean_distances
#from sklearn_extra.cluster import KMedoids

#X = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5])
#print(X.reshape(1,-1))
X = np.array([
                [1,0],
                [1,0],
                [1,0],
                [1,0],
                [1,0],
                [2,0],
                [2,0],
                [2,0],
                [2,0],
                [2,0],
                [3,0],
                [3,0],
                [3,0],
                [3,0],
                [3,0],
                [4,0],
                [4,0],
                [4,0],
                [4,0],
                [4,0],
                [5,0],
                [5,0],
                [5,0],
                [5,0],
                [5,0],
                [15,0]
                ])

myKMeans = KMeans(5)
ans = myKMeans.fit(X)
print(ans.cluster_centers_)

fittedX = []

print(ans.predict(X))

#ed = euclidean_distances(X)
#print(ed)
##print(np.__version__)
#
##print(X)
#
#kmedoidsObject = kmedoids.fastpam1(ed, 6)
#print(kmedoidsObject)
#for i in kmedoidsObject.medoids:
#    print(X[i])
#
##print(myMedians.cluster_centers)

#unsupervised Learning
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


#Attach Data Sets
DataSet=pd.read_csv(r'D:\ML\SKLearn Library\Slides && Data\Data\2.6 K-Means\data.csv')
X=DataSet.iloc[:500,:].values

ilist=[]
n=11
#n_clusters: Number of clusters by default =8
#init : init of center of every cluster (k-means++ , random)
# inertia_ : مجموع مربعات المسافات بين كل نقطة و مركزها , وكانها كوست فنكشن
#To show what is best number of cluster that make algorithm more optimal
for i in range(1,n):
    Kmeans_model=KMeans(n_clusters=i,init='k-means++',max_iter=10000,n_jobs=-1,algorithm='auto')
    Kmeans_model.fit(X)
    ilist.append(Kmeans_model.inertia_)

#Plot them to show what is best number of cluster that make algorithm more optimal
#We need to make inertia is more smaller
plt.plot(range(1,n),ilist)
plt.title('Elbow')
plt.xlabel('Clusters')
plt.ylabel('inertia')
plt.show()

#So from graph  best number of Clusters is 5 clusters
Kmeans_model=KMeans(n_clusters=5,init='k-means++',max_iter=10000,n_jobs=-1,algorithm='auto')
Y_Kmeans=Kmeans_model.fit_predict(X) # Fit and predict in one step

#Visualising The Clusters
plt.scatter(X[Y_Kmeans == 0, 0], X[Y_Kmeans == 0, 1], s = 10, c = 'r')
plt.scatter(X[Y_Kmeans == 1, 0], X[Y_Kmeans == 1, 1], s = 10, c = 'b')
plt.scatter(X[Y_Kmeans == 2, 0], X[Y_Kmeans == 2, 1], s = 10, c = 'g')
plt.scatter(X[Y_Kmeans == 3, 0], X[Y_Kmeans == 3, 1], s = 10, c = 'c')
plt.scatter(Kmeans_model.cluster_centers_[:, 0], Kmeans_model.cluster_centers_[:, 1], s = 100, c = 'y')
plt.show()

#------------------------------------
#another Example
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

Iris_data=load_iris()
X=Iris_data.data

"""
KMeansModel=KMeans(n_clusters=3)
KMeansModel.fit(X)

no_Cluster=KMeansModel.labels_
print("Name Of Clusters :",no_Cluster)
#Calc Score here
#print(silhouette_score(X,no_Cluster))
"""

Score=[]

#Get Best no Clusters that make algorithm more optimal
#the more Score is less the more algorithm is optimal so we should select min Score
for i in range(2,11):
    KMeansModel=KMeans(n_clusters=i)
    KMeansModel.fit(X)
    result=KMeansModel.labels_
    print(i,'    ' ,silhouette_score(X,result))
    Score.append(silhouette_score(X,result))


##Plot them to show what is best number of cluster that make algorithm more optimal

plt.plot(range(2,11),Score)
#plt.show()

#From graph no_clustrers = 6

KMeansModel = KMeans(n_clusters=6)
Y_Kmeans=KMeansModel.fit_predict(X)

#Visualising The Clusters

plt.scatter(X[Y_Kmeans == 0, 0], X[Y_Kmeans == 0, 1], s = 10, c = 'r')
plt.scatter(X[Y_Kmeans == 1, 0], X[Y_Kmeans == 1, 1], s = 10, c = 'b')
plt.scatter(X[Y_Kmeans == 2, 0], X[Y_Kmeans == 2, 1], s = 10, c = 'g')
plt.scatter(X[Y_Kmeans == 3, 0], X[Y_Kmeans == 3, 1], s = 10, c = 'c')

plt.scatter(X[Y_Kmeans == 4, 0], X[Y_Kmeans == 4, 1], s = 10, c = 'm')
plt.scatter(X[Y_Kmeans == 5, 0], X[Y_Kmeans == 5, 1], s = 10, c = 'y')
plt.scatter(KMeansModel.cluster_centers_[:, 0], KMeansModel.cluster_centers_[:, 1], s = 100, c = 'y')
plt.show()
#----------------------------------------
#MiniBatch K-Means  : ٍ

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs


# #############################################################################
# Generate sample data
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

# #############################################################################
# Compute clustering with Means

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

# #############################################################################
# Compute clustering with MiniBatchKMeans

mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0

# #############################################################################
# Plot result

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers,
                                  mbk_means_cluster_centers)

# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
    t_batch, k_means.inertia_))

# MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == order[k]
    cluster_center = mbk_means_cluster_centers[order[k]]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
         (t_mini_batch, mbk.inertia_))

# Initialise the different array to all False
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)

for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == order[k]))

identic = np.logical_not(different)
ax.plot(X[identic, 0], X[identic, 1], 'w',
        markerfacecolor='#bbbbbb', marker='.')
ax.plot(X[different, 0], X[different, 1], 'w',
        markerfacecolor='m', marker='.')
ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())

plt.show()

#-----------------------------------------------
#PCA
#المسؤله عن تقليل الابعاد في التعلم بدون اشراف
#PCA is responsibility of reduction and minimization  of Features(not delete Features but reduction Features) to make algorithm more optimal
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#n_components : number of Feature will remain هيدمج الفيوتشر مع بعض وعددهم هيبقي دا
#svd_solver : (auto,full,arpack,randomized
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)



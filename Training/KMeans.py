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







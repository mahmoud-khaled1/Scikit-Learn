from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#Load DataSets
Cancer=load_breast_cancer()
X=Cancer.data
Y=Cancer.target

#Splitting DataSets
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=44,shuffle=True)


#Apllying algorithm
#n_clusters:	عدد العناقيد
#affinity:	طريقة حساب المسافة بين النقاط , وتكون :  euclidean ,l1 ,l2 ,manhattan ,cosine ,precomputed
#linkage:	اسلوب الحل و تكون بين :   ward ,  complete ,  average ,  single
#tol:	السماحية المسموح بها
#n_jobs	 عدد المهام التي يتم تنفيذها بالتوازي
AgglomerativeClusteringModel=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

y_pred_train=AgglomerativeClusteringModel.fit_predict(X_train)
y_pred_test=AgglomerativeClusteringModel.fit_predict(X_test)

#draw the Hierarchical graph for Test set
dendrogram=sch.dendrogram(sch.linkage(X_test[:30,:],method='ward'))
plt.title('Test Set')
plt.xlabel('X Value')
plt.ylabel('Distance')
plt.show()

# draw the Scatter for Train set
plt.scatter(X_train[y_pred_train == 0, 0], X_train[y_pred_train == 0, 1], s=10, c='red', label='Cluster 1')
plt.scatter(X_train[y_pred_train == 1, 0], X_train[y_pred_train == 1, 1], s=10, c='blue', label='Cluster 2')
plt.scatter(X_train[y_pred_train == 2, 0], X_train[y_pred_train == 2, 1], s=10, c='green', label='Cluster 3')
plt.scatter(X_train[y_pred_train == 3, 0], X_train[y_pred_train == 3, 1], s=10, c='cyan', label='Cluster 4')
plt.scatter(X_train[y_pred_train == 4, 0], X_train[y_pred_train == 4, 1], s=10, c='magenta', label='Cluster 5')
plt.title('Training Set')
plt.xlabel('X Value')
plt.ylabel('y Value')
plt.legend()
plt.show()

# draw the Scatter for Test set
plt.scatter(X_test[y_pred_test == 0, 0], X_test[y_pred_test == 0, 1], s=10, c='red', label='Cluster 1')
plt.scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], s=10, c='blue', label='Cluster 2')
plt.scatter(X_test[y_pred_test == 2, 0], X_test[y_pred_test == 2, 1], s=10, c='green', label='Cluster 3')
plt.scatter(X_test[y_pred_test == 3, 0], X_test[y_pred_test == 3, 1], s=10, c='cyan', label='Cluster 4')
plt.scatter(X_test[y_pred_test == 4, 0], X_test[y_pred_test == 4, 1], s=10, c='magenta', label='Cluster 5')
plt.title('Testing Set')
plt.xlabel('X Value')
plt.ylabel('y Value')
plt.legend()
plt.show()



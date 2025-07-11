import math 

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

# data: pengeluaran pelanggan
data = [[100], [200], [220], [800], [850], [870]] 

kmeans = KMeans(n_clusters=2, random_state=0) 
kmeans.fit(data)

print(kmeans.labels_)  
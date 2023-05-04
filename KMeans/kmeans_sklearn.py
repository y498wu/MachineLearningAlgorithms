import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()

data = list(zip(x, y))
print(data)
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# By setting c to kmeans.labels_, 
# the scatter plot will color each data point based on its cluster label
plt.scatter(x, y, c=kmeans.labels_)
plt.show()
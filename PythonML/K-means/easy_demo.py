import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from KMeans_Normal_demo import SimpleKMeans

n_samples = 1500
random_state = 170
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X, y = make_blobs(n_samples=n_samples, random_state=random_state)

common_params = {"n_init": "auto", "random_state": random_state,}

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

#y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X)
model = SimpleKMeans(n_clusters=3, random_state=random_state)
#y_pred = model.labels
y_pred = model.fit(X)
y_pred = model.predict(X)

axs[0, 0].scatter(X[:, 0], X[:, 1], c=y_pred)
axs[0, 0].set_title("Non-optimal Number of Clusters")

plt.suptitle("Unexpected KMeans clusters").set_y(0.95)
plt.show()
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def read_csv(data):
    return pd.read_csv(data)


def scale(df):
    df = pd.get_dummies(df, dtype=int)
    scaler = StandardScaler()
    return scaler.fit_transform(df)


def elbow_plot(X, k_range=(1, 10), random_state=42):
    inertia = []
    k_values = list(range(k_range[0], k_range[1]))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X)
        inertia.append(kmeans.inertia_)

    table = wandb.Table(["k", "inertia"])
    for k, i in zip(k_values, inertia):
        table.add_data(k, i)
    wandb.log({"elbow_table": table})

    plt.plot(inertia)
    plt.title("Elbow Method")
    plt.xlabel("k-Number of Cluster")
    plt.ylabel("Percentage of Variance Explained")
    wandb.log({"elbow_plot": wandb.Image(plt.gcf())})
    plt.close()


def kmeans(X, n_clusters=4, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)

    scatter = plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
    plt.title("KMeans Clustering of Penguins")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    wandb.log({"kmeans_plot": wandb.Image(plt.gcf())})
    plt.close()

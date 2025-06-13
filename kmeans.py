import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


class CustomKMeans:
    def __init__(self, n_clusters=3, max_iterations=300, random_seed=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.centers = None
        self.cluster_labels = None

    def initialize_centers(self, data):
        np.random.seed(self.random_seed)
        random_indices = np.random.permutation(data.shape[0])[:self.n_clusters]
        return data[random_indices]

    def compute_distances(self, data, centers): # евк расст
        distances = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
        return distances

    def update_centers(self, data, labels): # среднее всех точек
        new_centers = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            new_centers[i] = np.mean(data[labels == i], axis=0)
        return new_centers

    def plot_clusters(self, data, iteration): # наст класт
        plt.figure(figsize=(8, 6))
        colors = ['red', 'green', 'blue', 'purple', 'orange'][:self.n_clusters]

        for i in range(self.n_clusters):
            cluster_data = data[self.cluster_labels == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                        color=colors[i], label=f'Cluster {i + 1}')

        plt.scatter(self.centers[:, 0], self.centers[:, 1],
                    color='black', marker='X', s=200, label='Centers')
        plt.title(f'Iteration {iteration}')
        plt.legend()
        plt.show()

    def fit(self, data, plot_steps=True): # осн алг
        self.centers = self.initialize_centers(data)

        for iteration in range(self.max_iterations):
            distances = self.compute_distances(data, self.centers)
            self.cluster_labels = np.argmin(distances, axis=1)

            if plot_steps:
                self.plot_clusters(data, iteration + 1)

            new_centers = self.update_centers(data, self.cluster_labels)

            if np.allclose(self.centers, new_centers): # пров сход
                break

            self.centers = new_centers


def find_best_cluster_count(data, max_clusters=5):
    distortions = []
    for k in range(1, max_clusters + 1):
        model = CustomKMeans(n_clusters=k)
        model.fit(data, plot_steps=False)
        # s расст кв от . до ц
        distances = np.linalg.norm(data - model.centers[model.cluster_labels], axis=1)
        distortions.append(np.sum(distances))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), distortions, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal Cluster Count')
    plt.show()

    diffs = np.diff(distortions)
    relative_diffs = diffs[1:] / diffs[:-1]
    optimal_k = np.argmin(relative_diffs) + 2
    return optimal_k


def main():
    iris = load_iris()
    X = iris.data
    # 2d c PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    optimal_k = find_best_cluster_count(X_pca)
    print(f"Optimal number of clusters: {optimal_k}")

    kmeans = CustomKMeans(n_clusters=optimal_k, random_seed=42)
    kmeans.fit(X_pca, plot_steps=True)

    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue', 'purple', 'orange'][:optimal_k]

    for i in range(optimal_k):
        cluster_data = X_pca[kmeans.cluster_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    color=colors[i], label=f'Cluster {i + 1}')

    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1],
                color='black', marker='X', s=200, label='Final Centers')
    plt.title('Final Clustering Result')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

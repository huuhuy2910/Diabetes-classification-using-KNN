import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi

np.random.seed(4)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal(means[0], cov, 500)
X1 = np.random.multivariate_normal(means[1], cov, 500)
X2 = np.random.multivariate_normal(means[2], cov, 500)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k)]

def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis=1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis=0)
    return centers

def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    max_it = 6
    it = 0
    while it < max_it:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print(centers[-1])
print(labels[-1].shape)

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1] 
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

fig, ax = plt.subplots()

def update(ii):
    label2 = 'iteration {0}: '.format(int(ii/2))
    if ii % 2:
        label2 += ' update centers'
    else:
        label2 += ' assign points to clusters'

    i_c = int((ii+1)/2)
    i_p = int(ii/2)

    label = labels[int(i_p)]
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.cla()
    plt.axis('equal')
    plt.axis([-2, 12, -3, 12])

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    i = int(i_c)
    plt.plot(centers[i][0, 0], centers[i][0, 1], 'y^', markersize=15)
    plt.plot(centers[i][1, 0], centers[i][1, 1], 'yo', markersize=15)
    plt.plot(centers[i][2, 0], centers[i][2, 1], 'ys', markersize=15)

    points = centers[i]
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=1000)
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=.2)

    ax.set_xlabel(label2)
    return ax

anim = FuncAnimation(fig, update, frames=np.arange(0, 2*it), interval=1000)
plt.show()

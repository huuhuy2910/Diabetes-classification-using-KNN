from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Khởi tạo ngẫu nhiên các tham số
np.random.seed(11)
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500  # Số lượng điểm dữ liệu
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# Kết hợp dữ liệu thành một mảng duy nhất
X = np.concatenate((X0, X1, X2), axis=0)
K = 3  # Số lượng cụm

# Nhãn gốc cho dữ liệu (để so sánh)
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# Hàm hiển thị dữ liệu và nhãn
def kmeans_display(X, label):
    K = np.amax(label) + 1  # Số lượng cụm
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()

# Hiển thị dữ liệu ban đầu
kmeans_display(X, original_label)

# Hàm khởi tạo tâm ngẫu nhiên
def kmeans_init_centers(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]

# Hàm gán nhãn cho các điểm dữ liệu
def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)  # Tính khoảng cách giữa các điểm dữ liệu và tâm
    return np.argmin(D, axis=1)  # Trả về nhãn dựa trên khoảng cách nhỏ nhất

# Hàm cập nhật tâm mới dựa trên nhãn
def kmeans_update_centers(X, labels, K):
    new_centers = np.zeros((K, X.shape[1]))  # Khởi tạo mảng cho các tâm mới
    for k in range(K):
        Xk = X[labels == k, :]  # Lấy các điểm thuộc cụm k
        # Cập nhật tâm là trung bình của các điểm thuộc cụm k
        new_centers[k, :] = np.mean(Xk, axis=0) if len(Xk) > 0 else np.zeros((X.shape[1],))
    return new_centers

# Hàm kiểm tra hội tụ
def has_converged(centers, new_centers):
    return np.array_equal(centers, new_centers)

# Hàm thực hiện thuật toán K-means
def kmeans(X, K):
    # Khởi tạo tâm ban đầu bằng cách chọn ngẫu nhiên K điểm từ dữ liệu
    centers = [kmeans_init_centers(X, K)]
    labels = []  # Danh sách lưu nhãn cho từng điểm dữ liệu
    max_it = 100  # Giới hạn số lần lặp tối đa
    it = 0  # Biến đếm số lần lặp

    # Vòng lặp chính cho thuật toán K-means
    while it < max_it:
        # Gán nhãn cho các điểm dữ liệu dựa trên tâm hiện tại
        labels.append(kmeans_assign_labels(X, centers[-1]))
        
        # Cập nhật vị trí các tâm dựa trên nhãn mới
        new_centers = kmeans_update_centers(X, labels[-1], K)
        
        # Kiểm tra xem các tâm có hội tụ hay không
        if has_converged(centers[-1], new_centers):
            break  # Nếu hội tụ, dừng vòng lặp
            
        centers.append(new_centers)  # Cập nhật danh sách các tâm
        it += 1  # Tăng số lần lặp

    # Trả về các tâm, nhãn và số lần lặp thực tế
    return (centers, labels, it)

# Chạy thuật toán K-means
centers, labels, it = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])  # In tâm cuối cùng được tìm thấy

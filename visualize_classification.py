import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

def plot_step(X, y, title, filename=None):
    """Helper function để plot từng bước"""
    plt.figure(figsize=(6, 4))
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    
    # Nếu dữ liệu có nhiều hơn 2 chiều, chỉ plot 2 chiều đầu tiên
    if X.shape[1] > 2:
        X_plot = X[:, :2]
    else:
        X_plot = X
        
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=cm_bright)
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_decision_boundary(X, y, clf, ax, title):
    """Plot decision boundary và dữ liệu"""
    h = 0.02  # step size in the mesh
    
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    # Plot dữ liệu train/test riêng biệt
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    plt.title("Training Data")
    
    plt.subplot(122)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright)
    plt.title("Testing Data")
    
    plt.savefig(f'step_train_test_split_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Train model
    clf.fit(X_train, y_train)
    
    # Tạo lưới điểm để vẽ decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Tạo colormap
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    # Plot lưới điểm
    plt.figure(figsize=(6, 4))
    plt.scatter(xx.ravel(), yy.ravel(), c='lightgray', alpha=0.1, s=1)
    plt.title("Mesh Grid Points")
    plt.savefig(f'step_mesh_grid_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Tính decision boundary
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

    # Plot decision boundary
    Z = Z.reshape(xx.shape)
    
    # Plot riêng decision boundary
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    plt.title("Decision Boundary")
    plt.savefig(f'step_decision_boundary_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot tổng hợp trên axes chính
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    
    # Plot training points
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=cm_bright,
        edgecolors="black",
        s=25,
    )
    # Plot testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cm_bright,
        alpha=0.6,
        edgecolors="black",
        s=25,
    )

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    # Tính và hiển thị độ chính xác
    score = clf.score(X_test, y_test)
    ax.set_title(title)
    ax.text(
        xx.max() - 0.3,
        yy.min() + 0.3,
        f"{score:.3f}".lstrip("0"),
        size=15,
        horizontalalignment="right",
    )

def visualize_classification(data_path="merged_data_1_5_18.csv"):
    # Load dữ liệu
    data = pd.read_csv(data_path)
    X = data.drop(columns=['state'])
    y = data["state"].copy()  # Tạo bản sao để tránh warning
    
    # Chuyển đổi nhãn thành số ngay từ đầu
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Plot dữ liệu gốc (chỉ 2 feature đầu tiên)
    plot_step(X.values, y, "Original High-dimensional Data (First 2 Features)", "step1_original_data.png")
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Plot sau khi scale
    plot_step(X_scaled, y, "Scaled Data (First 2 Features)", "step2_scaled_data.png")
    
    # Giảm chiều dữ liệu xuống 2D để visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Plot sau khi PCA
    plot_step(X_2d, y, "2D Data after PCA", "step3_pca_data.png")
    
    # Thiết lập các giá trị alpha khác nhau
    alphas = np.logspace(-3, 1, 5)  # [0.001, 0.01, 0.1, 1.0, 10.0]
    
    # Tạo figure
    figure = plt.figure(figsize=(17, 4))
    
    # Plot dữ liệu gốc
    ax = plt.subplot(1, len(alphas) + 1, 1)
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    
    # Chia dữ liệu train/test cho plot đầu tiên
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.4, random_state=42
    )
    
    # Plot training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # Plot testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Input data")
    
    # Train và plot với các giá trị alpha khác nhau
    for idx, alpha in enumerate(alphas):
        model = MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=800,
            alpha=alpha,
            activation="tanh",
            solver="adam",
            learning_rate_init=0.0005,
            random_state=42
        )
        
        ax = plt.subplot(1, len(alphas) + 1, idx + 2)
        plot_decision_boundary(X_2d, y, model, ax, f"alpha {alpha:.3f}")
    
    plt.tight_layout()
    plt.savefig('final_classification_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_classification() 
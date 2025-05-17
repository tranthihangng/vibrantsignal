import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_features(data_path="merged_data_finalno_fail.csv"):
    # Load dữ liệu
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    # Phân tích cơ bản
    print("\n=== Thông tin cơ bản ===")
    print(f"Tổng số mẫu: {len(data)}")
    print("\nPhân bố nhãn:")
    print(data['state'].value_counts())
    
    # Tách dữ liệu theo nhãn
    normal_data = data[data['state'] == 'normal1'].drop('state', axis=1)
    fault_data = data[data['state'] == 'rung5_18'].drop('state', axis=1)
    
    # Thống kê cho từng nhóm
    print("\n=== Thống kê cho dữ liệu NORMAL ===")
    print("\nSố lượng mẫu:", len(normal_data))
    print("\nThống kê cơ bản:")
    print(normal_data.describe())
    
    print("\n=== Thống kê cho dữ liệu FAULT ===")
    print("\nSố lượng mẫu:", len(fault_data))
    print("\nThống kê cơ bản:")
    print(fault_data.describe())
    
    # Vẽ boxplot cho mỗi feature
    plt.figure(figsize=(20, 10))
    plt.title("Phân bố các đặc trưng theo nhãn")
    
    # Chuyển dữ liệu về dạng phù hợp cho boxplot
    melted_data = pd.melt(data, id_vars=['state'], var_name='Feature', value_name='Value')
    sns.boxplot(x='Feature', y='Value', hue='state', data=melted_data)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
    
    # Vẽ histogram cho mỗi feature
    features = data.drop('state', axis=1).columns
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5*n_rows))
    
    for idx, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, idx)
        plt.title(f'Distribution of {feature}')
        sns.histplot(data=data, x=feature, hue='state', multiple="layer", alpha=0.5)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_histograms.png', dpi=300, bbox_inches='tight')
    
    # Tính correlation matrix cho mỗi nhóm
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.title("Correlation Matrix - Normal Data")
    sns.heatmap(normal_data.corr(), cmap='RdBu', center=0, annot=False)
    
    plt.subplot(122)
    plt.title("Correlation Matrix - Fault Data")
    sns.heatmap(fault_data.corr(), cmap='RdBu', center=0, annot=False)
    
    plt.tight_layout()
    plt.savefig('correlation_matrices.png', dpi=300, bbox_inches='tight')
    
    # Tính và in ra các feature có sự khác biệt lớn nhất giữa hai nhóm
    print("\n=== Feature Importance based on difference ===")
    feature_diff = {}
    for feature in features:
        normal_mean = normal_data[feature].mean()
        fault_mean = fault_data[feature].mean()
        normal_std = normal_data[feature].std()
        fault_std = fault_data[feature].std()
        
        # Tính effect size (Cohen's d)
        pooled_std = np.sqrt((normal_std**2 + fault_std**2) / 2)
        effect_size = abs(normal_mean - fault_mean) / pooled_std
        
        feature_diff[feature] = effect_size
    
    # Sắp xếp và in ra top features
    sorted_features = sorted(feature_diff.items(), key=lambda x: x[1], reverse=True)
    print("\nTop features phân biệt giữa normal và fault (dựa trên Cohen's d):")
    for feature, score in sorted_features[:10]:
        print(f"{feature}: {score:.3f}")
        print(f"  Normal - Mean: {normal_data[feature].mean():.3f}, Std: {normal_data[feature].std():.3f}")
        print(f"  Fault  - Mean: {fault_data[feature].mean():.3f}, Std: {fault_data[feature].std():.3f}")

if __name__ == "__main__":
    analyze_features() 
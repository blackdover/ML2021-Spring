import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def analyze_data():
    """
    对TIMIT数据集进行初步的数据分析和特征探索。
    1. 检查并可视化类别分布，判断数据是否存在不均衡问题。
    2. 使用PCA进行降维，并将高维特征投影到二维空间进行可视化。
    3. 使用随机森林模型来评估各个特征的重要性。
    """
    print("--- 开始数据分析 ---")
    
    # --- 1. 数据加载 ---
    print("正在加载数据...")
    data_root = './timit_11/'
    try:
        train_data = np.load(data_root + 'train_11.npy')
        train_label = np.load(data_root + 'train_label_11.npy')
    except FileNotFoundError:
        print(f"错误：请确保 {data_root} 目录下存在 train_11.npy 和 train_label_11.npy 文件。")
        return

    print(f"训练数据大小: {train_data.shape}")
    print(f"训练标签大小: {train_label.shape}")

    # --- 2. 类别分布分析 ---
    print("\n--- 正在分析类别分布 ---")
    plt.figure(figsize=(15, 6))
    sns.countplot(x=train_label)
    plt.title('Distribution of Phoneme Classes in Training Data')
    plt.xlabel('Phoneme Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print("类别分布图已保存至 class_distribution.png")
    plt.close()
    
    # 检查是否有严重不均衡
    class_counts = pd.Series(train_label).value_counts()
    print("各类别样本数量 (前5):")
    print(class_counts.head())
    if class_counts.min() < class_counts.mean() * 0.2:
        print("\n警告：存在潜在的类别不均衡问题，某些类别的样本数量远低于平均值。")

    # --- 3. PCA降维与可视化 ---
    # 由于数据量大，我们取一部分样本进行可视化，否则会非常慢且图会很拥挤
    print("\n--- 正在进行PCA降维与可视化 (使用10000个样本) ---")
    sample_indices = np.random.choice(train_data.shape[0], 10000, replace=False)
    sample_data = train_data[sample_indices]
    sample_labels = train_label[sample_indices]

    # 归一化是PCA的前提
    scaler = StandardScaler()
    scaled_sample_data = scaler.fit_transform(sample_data)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_sample_data)
    
    print(f"PCA解释的方差比例: {pca.explained_variance_ratio_}")
    print(f"PCA累计解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=sample_labels.astype(int), cmap='viridis', s=10, alpha=0.7)
    plt.title('2D PCA of TIMIT Data (10k Samples)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(39)), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('pca_visualization.png')
    print("PCA可视化图已保存至 pca_visualization.png")
    plt.close()

    # --- 4. 特征重要性分析 (使用随机森林) ---
    print("\n--- 正在使用随机森林进行特征重要性分析 (使用20000个样本) ---")
    start_time_rf = time.time()
    
    rf_sample_indices = np.random.choice(train_data.shape[0], 20000, replace=False)
    rf_sample_data = train_data[rf_sample_indices]
    rf_sample_labels = train_label[rf_sample_indices]

    # 训练随机森林模型
    # n_jobs=-1 使用所有可用CPU核心
    forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    forest.fit(rf_sample_data, rf_sample_labels)
    
    end_time_rf = time.time()
    print(f"随机森林训练耗时: {(end_time_rf - start_time_rf):.2f} 秒")
    
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 打印最重要的30个特征
    # 打印所有特征的重要性
    N_FEATURES = len(importances)
    print(f"\n所有 {N_FEATURES} 个特征的重要性:")
    for i in range(N_FEATURES):
        print(f"{i + 1}. 特征 {indices[i]} ({importances[indices[i]]:.4f})")
    
    # 保存所有特征重要性到CSV文件
    import pandas as pd
    feature_importance_df = pd.DataFrame({
        'Feature_Index': indices,
        'Importance': importances[indices]
    })
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("所有特征重要性已保存至 feature_importance.csv")

    # 可视化特征重要性
    plt.figure(figsize=(15, 6))
    plt.title(f"Top {N_FEATURES} Feature Importances")
    plt.bar(range(N_FEATURES), importances[indices][:N_FEATURES], align='center')
    plt.xticks(range(N_FEATURES), indices[:N_FEATURES])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print(f"\n特征重要性图已保存至 feature_importance.png")
    plt.close()

    print("\n--- 数据分析完成 ---")

if __name__ == '__main__':
    analyze_data()

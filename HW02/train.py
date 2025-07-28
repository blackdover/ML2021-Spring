import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import gc
import os
import copy
import pickle
import time

# ==============================================================================
#                             --- 配置参数 ---
# ==============================================================================

# 数据相关配置
VAL_RATIO = 0.1              # 验证集比例（仅在非交叉验证时使用）
BATCH_SIZE = 256             # 批处理大小
N_FOLDS = 3                  # 交叉验证折数
USE_CV = False               # 是否使用交叉验证 (True: K-Fold, False: Train/Val split)
 
# 随机种子配置
SEED = 182142                # 随机种子值

# 特征选择与预处理配置
N_FEATURES = 200             # 选择的特征数量
USE_FEATURE_SELECTION = False # 是否进行特征选择
USE_NORMALIZATION = True     # 是否进行归一化

# 训练相关配置
NUM_EPOCH = 200              # 最大训练轮数
LEARNING_RATE = 0.00005       # 学习率
WEIGHT_DECAY = 1e-5          # L2正则化系数
MODEL_PATH = './model.ckpt'  # 模型保存路径
RESUME_TRAINING = True      # 是否从断点继续训练
SELECTOR_PATH = './feature_selector.pkl' # 特征选择器保存路径
SCALER_PATH = './scaler.pkl'           # 归一化器保存路径

# 学习率调度器配置
USE_SCHEDULER = True         # 是否使用学习率调度器
SCHEDULER_PATIENCE = 5       # 学习率调度器的耐心值
SCHEDULER_FACTOR = 0.005       # 学习率缩放因子 (new_lr = old_lr * factor)

# 早停配置
EARLY_STOPPING_PATIENCE = 10 # 早停的耐心值

# ==============================================================================
#                             --- 辅助函数 ---
# ==============================================================================

def get_device():
  """获取可用的计算设备 (GPU或CPU)"""
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def same_seeds(seed):
    """固定随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ==============================================================================
#                             --- 数据集类 ---
# ==============================================================================

class TIMITDataset(Dataset):
    """自定义的TIMIT语音数据集"""
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

# ==============================================================================
#                              --- 模型类 ---
# ==============================================================================

class Classifier(nn.Module):
    """神经网络分类器模型"""
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.4),
            
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.4),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.4),

            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.4),
            
            nn.Linear(256, 39)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)

# ==============================================================================
#                             --- 主函数 ---
# ==============================================================================

def main():
    """主执行函数"""
    # --- 初始化 ---
    same_seeds(SEED)
    device = get_device()
    print(f"使用设备: {device}")
    start_time = time.time()

    # --- 数据加载 ---
    print("正在加载数据...")
    data_root = './timit_11/'
    train_data = np.load(data_root + 'train_11.npy')
    train_label = np.load(data_root + 'train_label_11.npy')
    test_data = np.load(data_root + 'test_11.npy')
    print(f"原始训练数据大小: {train_data.shape}")
    print(f"原始测试数据大小: {test_data.shape}")

    # --- 数据预处理 ---
    input_dim = train_data.shape[1]

    if USE_FEATURE_SELECTION:
        should_refit_selector = True
        if os.path.exists(SELECTOR_PATH):
            print(f"正在从 {SELECTOR_PATH} 加载特征选择器...")
            with open(SELECTOR_PATH, 'rb') as f:
                selector = pickle.load(f)
            # 检查选择器的k值是否与配置匹配
            if selector.k == N_FEATURES:
                print(f"特征选择器参数 (k={selector.k}) 与配置 (N_FEATURES={N_FEATURES}) 匹配，将直接使用。")
                should_refit_selector = False
            else:
                print(f"特征选择器参数 (k={selector.k}) 与配置 (N_FEATURES={N_FEATURES}) 不匹配，将重新训练。")

        if should_refit_selector:
            print(f"正在选择最重要的 {N_FEATURES} 个特征...")
            selector = SelectKBest(f_classif, k=N_FEATURES)
            selector.fit(train_data, train_label)
            with open(SELECTOR_PATH, 'wb') as f:
                pickle.dump(selector, f)
            print(f"新的特征选择器已保存至: {SELECTOR_PATH}")

        train_data = selector.transform(train_data)
        test_data = selector.transform(test_data)
        input_dim = train_data.shape[1]
        print(f"特征选择后训练数据大小: {train_data.shape}")

    if USE_NORMALIZATION:
        should_refit_scaler = True
        if os.path.exists(SCALER_PATH):
            print(f"正在从 {SCALER_PATH} 加载归一化器...")
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            
            if scaler.n_features_in_ == train_data.shape[1]:
                print(f"归一化器特征维度 ({scaler.n_features_in_}) 与当前数据维度 ({train_data.shape[1]}) 匹配，将直接使用。")
                should_refit_scaler = False
            else:
                print(f"归一化器特征维度 ({scaler.n_features_in_}) 与当前数据维度 ({train_data.shape[1]}) 不匹配，将重新训练。")
        
        if should_refit_scaler:
            print("正在进行数据归一化...")
            scaler = StandardScaler()
            scaler.fit(train_data)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"新的归一化器已保存至: {SCALER_PATH}")

        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        print("归一化完成")


    # --- 训练模型 ---
    if USE_CV:
        print("\n--- 开始进行K折交叉验证 ---")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_val_accs = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            print(f"\n=== 第 {fold+1}/{N_FOLDS} 折训练开始 ===")
            
            train_x, train_y = train_data[train_idx], train_label[train_idx]
            val_x, val_y = train_data[val_idx], train_label[val_idx]
            
            train_set = TIMITDataset(train_x, train_y)
            val_set = TIMITDataset(val_x, val_y)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
            
            model = Classifier(input_dim=input_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, 'max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE) if USE_SCHEDULER else None

            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(NUM_EPOCH):
                model.train()
                train_loss, train_acc = 0.0, 0.0
                for data_batch in train_loader:
                    inputs, labels = data_batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = model.cal_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    _, pred = torch.max(outputs, 1)
                    train_acc += (pred.cpu() == labels.cpu()).sum().item()
                    train_loss += loss.item()
                
                model.eval()
                val_loss, val_acc = 0.0, 0.0
                with torch.no_grad():
                    for data_batch in val_loader:
                        inputs, labels = data_batch
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = model.cal_loss(outputs, labels)
                        _, pred = torch.max(outputs, 1)
                        val_acc += (pred.cpu() == labels.cpu()).sum().item()
                        val_loss += loss.item()
                
                avg_val_acc = val_acc / len(val_set)
                print(f"[折 {fold+1}, 轮 {epoch+1:03d}] 训练损失: {train_loss/len(train_loader):.4f}, 验证准确率: {avg_val_acc:.4f}")

                if scheduler:
                    scheduler.step(avg_val_acc)
                
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    torch.save(model.state_dict(), f'./model_fold_{fold+1}.ckpt')
                    print(f"  -> 保存第 {fold+1} 折最佳模型，准确率: {best_val_acc:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"  -> 验证准确率连续 {EARLY_STOPPING_PATIENCE} 轮未提升，提前停止。")
                    break
            
            fold_val_accs.append(best_val_acc)
        
        print(f"\n交叉验证完成。平均验证准确率: {np.mean(fold_val_accs):.4f}")

        # 在全部数据上重新训练最终模型
        print("\n--- 在全部训练数据上训练最终模型 ---")
        full_train_set = TIMITDataset(train_data, train_label)
        full_train_loader = DataLoader(full_train_set, batch_size=BATCH_SIZE, shuffle=True)
        model = Classifier(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        for epoch in range(NUM_EPOCH // 2): # 在全数据上训练一半的轮数
            model.train()
            for data_batch in full_train_loader:
                inputs, labels = data_batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.cal_loss(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"[最终模型 轮 {epoch+1:03d}] 训练中...")
        
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"最终模型已保存至 {MODEL_PATH}")

    else: # 使用常规训练-验证集模式
        print("\n--- 开始常规训练 (Train/Validation Split) ---")
        
        percent = int(train_data.shape[0] * (1 - VAL_RATIO))
        train_x, train_y = train_data[:percent], train_label[:percent]
        val_x, val_y = train_data[percent:], train_label[percent:]
        
        train_set = TIMITDataset(train_x, train_y)
        val_set = TIMITDataset(val_x, val_y)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
        
        model = Classifier(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        best_val_acc = 0.0
        patience_counter = 0

        if RESUME_TRAINING and os.path.exists(MODEL_PATH):
            print(f"从 {MODEL_PATH} 加载模型断点继续训练...")
            model.load_state_dict(torch.load(MODEL_PATH))
            print("模型加载成功，正在评估当前验证集准确率...")
            model.eval()
            val_acc = 0.0
            with torch.no_grad():
                for data_batch in val_loader:
                    inputs, labels = data_batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    val_acc += (pred.cpu() == labels.cpu()).sum().item()
            best_val_acc = val_acc / len(val_set)
            print(f"加载模型的验证准确率为: {best_val_acc:.4f}，将从这里开始继续提升。")
            
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE) if USE_SCHEDULER else None
        
        for epoch in range(NUM_EPOCH):
            model.train()
            train_loss, train_acc = 0.0, 0.0
            for data_batch in train_loader:
                inputs, labels = data_batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.cal_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                _, pred = torch.max(outputs, 1)
                train_acc += (pred.cpu() == labels.cpu()).sum().item()
                train_loss += loss.item()

            model.eval()
            val_loss, val_acc = 0.0, 0.0
            with torch.no_grad():
                for data_batch in val_loader:
                    inputs, labels = data_batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = model.cal_loss(outputs, labels)
                    _, pred = torch.max(outputs, 1)
                    val_acc += (pred.cpu() == labels.cpu()).sum().item()
                    val_loss += loss.item()
            
            avg_val_acc = val_acc / len(val_set)
            print(f"[轮 {epoch+1:03d}] 训练损失: {train_loss/len(train_loader):.4f}, 验证准确率: {avg_val_acc:.4f}")

            if scheduler:
                scheduler.step(avg_val_acc)
            
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  -> 保存最佳模型，准确率: {best_val_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  -> 验证准确率连续 {EARLY_STOPPING_PATIENCE} 轮未提升，提前停止。")
                break
    
    # --- 测试 ---
    print("\n--- 开始测试 ---")
    del train_data, train_label
    gc.collect()

    test_set = TIMITDataset(test_data, None)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"加载模型 {MODEL_PATH} 用于预测...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    predict = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            predict.extend(pred.cpu().numpy())

    # --- 生成提交文件 ---
    print("正在生成提交文件 prediction.csv...")
    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(predict):
            f.write(f'{i},{y}\n')
    
    print("完成！")
    end_time = time.time()
    print(f"总耗时: {((end_time - start_time) / 60):.2f} 分钟")


if __name__ == "__main__":
    main()

import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader

# ======= 从notebook复制的数据集类 =======
class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        # 初始化方法接收特征数据X和可选的标签数据y
        self.data = torch.from_numpy(X).float()
        if y is not None:
            # 如果提供了标签数据y
            # 将y转换为整数类型
            y = y.astype(int)
            # 然后转换为PyTorch长整型张量
            self.label = torch.LongTensor(y)
        else:
            # 如果没有提供标签数据，则标签设为None（用于测试集）
            self.label = None

    def __getitem__(self, idx):
        # 获取指定索引的数据项
        if self.label is not None:
            # 如果有标签，返回(数据,标签)对
            return self.data[idx], self.label[idx]
        else:
            # 如果没有标签，只返回数据
            return self.data[idx]

    def __len__(self):
        # 返回数据集的大小（样本数量）
        return len(self.data)

# ======= 从notebook复制的模型类 =======
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim=200):  # 默认使用200个特征
        super(Classifier, self).__init__()
        # 定义神经网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.2),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            
            nn.Linear(128, 39)  # 输出39个类别
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)

# ======= 特征选择和归一化工具 =======
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def validate_model():
    # 配置参数
    MODEL_PATH = './model.ckpt'           # 默认模型路径
    DATA_ROOT = './timit_11/'            # 数据根目录
    BATCH_SIZE = 256                    # 批处理大小
    N_FEATURES = 200                    # 特征数量（明确设置为200，与训练时一致）
    FEATURE_SELECTOR_PATH = './feature_selector.pkl'  # 特征选择器保存路径
    SCALER_PATH = './scaler.pkl'        # 标准化器保存路径
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载验证数据
    print("加载验证数据...")
    try:
        val_data = np.load(DATA_ROOT + 'val_11.npy')
        val_label = np.load(DATA_ROOT + 'val_label_11.npy')
        print(f"验证数据大小: {val_data.shape}")
        print(f"验证标签大小: {val_label.shape}")
    except Exception as e:
        print(f"加载验证数据失败: {e}")
        return
    
    # 特征选择流程
    selector = None
    if os.path.exists(FEATURE_SELECTOR_PATH):
        print(f"加载已保存的特征选择器: {FEATURE_SELECTOR_PATH}")
        try:
            with open(FEATURE_SELECTOR_PATH, 'rb') as f:
                selector = pickle.load(f)
            print("特征选择器加载成功")
        except Exception as e:
            print(f"加载特征选择器失败: {e}")
            selector = None
    
    # 如果没有加载到特征选择器，训练一个新的
    if selector is None:
        print(f"创建新的特征选择器 (k={N_FEATURES})...")
        try:
            # 加载训练数据
            train = np.load(DATA_ROOT + 'train_11.npy')
            train_label = np.load(DATA_ROOT + 'train_label_11.npy')
            
            # 创建并训练选择器
            selector = SelectKBest(f_classif, k=N_FEATURES)
            selector.fit(train, train_label)
            
            # 保存特征选择器
            with open(FEATURE_SELECTOR_PATH, 'wb') as f:
                pickle.dump(selector, f)
            print(f"特征选择器已保存至: {FEATURE_SELECTOR_PATH}")
            
            # 释放内存
            del train, train_label
        except Exception as e:
            print(f"创建特征选择器失败: {e}")
            return
    
    # 应用特征选择
    print("应用特征选择到验证数据...")
    val_data = selector.transform(val_data)
    print(f"特征选择后验证数据大小: {val_data.shape}")
    
    # 标准化流程
    scaler = None
    if os.path.exists(SCALER_PATH):
        print(f"加载已保存的标准化器: {SCALER_PATH}")
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("标准化器加载成功")
        except Exception as e:
            print(f"加载标准化器失败: {e}")
            scaler = None
    
    # 如果没有加载到标准化器，创建一个新的
    if scaler is None:
        print("创建新的标准化器...")
        scaler = StandardScaler()
        
        # 对验证数据进行拟合和转换
        val_data = scaler.fit_transform(val_data)
        
        # 保存标准化器
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"标准化器已保存至: {SCALER_PATH}")
    else:
        # 使用加载的标准化器转换验证数据
        val_data = scaler.transform(val_data)
    
    print("验证数据预处理完成")
    
    # 创建验证数据集和数据加载器
    val_set = TIMITDataset(val_data, val_label)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型 - 使用200个特征
    print(f"创建模型，输入维度: {N_FEATURES}")
    model = Classifier(input_dim=N_FEATURES).to(device)
    
    # 尝试加载默认模型
    if os.path.exists(MODEL_PATH):
        print(f"尝试加载默认模型: {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"成功加载默认模型")
        except Exception as e:
            print(f"加载默认模型失败: {e}")
            print("无法加载模型，退出验证")
            return
    else:
        print(f"模型文件不存在: {MODEL_PATH}")
        return
    
    # 设置为评估模式
    model.eval()
    
    # 进行验证
    print("开始进行模型验证...")
    correct = 0
    total = 0
    val_loss = 0.0
    
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = model.cal_loss(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算准确率和平均损失
    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    
    print(f"验证完成，总样本数: {total}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"平均损失: {avg_loss:.4f}")

if __name__ == "__main__":
    validate_model()

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomGMM(nn.Module):
    def __init__(self, n_components, n_features, covariance_type='diag'):
        super(CustomGMM, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.covariance_type = covariance_type
        
        # 初始化参数
        self.mus = nn.Parameter(torch.randn(n_components, n_features))  # 聚类中心
        self.log_pi = nn.Parameter(torch.zeros(n_components))           # 混合系数的对数
        
        # 协方差参数初始化（根据协方差类型）
        if covariance_type == 'diag':
            self.log_vars = nn.Parameter(torch.zeros(n_components, n_features))
        elif covariance_type == 'spherical':
            self.log_vars = nn.Parameter(torch.zeros(n_components, 1))
        else:
            raise ValueError("covariance_type 必须是 'diag' 或 'spherical'")
    
    def forward(self, X):
        """
        计算每个样本对每个成分的负对数似然（损失函数）
        """
        # 计算自定义距离（例如马氏距离）
        dist = self._mahalanobis_distance(X)
        
        # 计算对数概率密度
        log_prob = self._compute_log_prob(dist)
        
        # 混合系数的对数加上各成分的对数概率
        weighted_log_prob = self.log_pi.unsqueeze(0) + log_prob
        
        # 对数似然：对数求和指数技巧
        log_likelihood = torch.logsumexp(weighted_log_prob, dim=1)
        
        # 损失为负对数似然的均值
        loss = -log_likelihood.mean()
        return loss
    
    def _mahalanobis_distance(self, X):
        """
        自定义马氏距离（假设协方差为对角矩阵）
        """
        # X形状: (batch_size, n_features)
        # mus形状: (n_components, n_features)
        # log_vars形状: (n_components, n_features) 或 (n_components, 1)
        variances = torch.exp(self.log_vars)
        
        # 计算马氏距离：sum((X - mu)^2 / variance)
        diff = X[0].unsqueeze(1) - self.mus.unsqueeze(0)  # (batch, n_components, n_features)
        dist = (diff ** 2) / variances.unsqueeze(0)
        dist = dist.sum(dim=-1)  # (batch, n_components)
        return dist
    
    def _compute_log_prob(self, dist):
        """
        根据距离计算对数概率密度
        """
        # 协方差矩阵的对数行列式（对角协方差）
        log_det = torch.sum(self.log_vars, dim=-1)  # (n_components,)
        
        # 对数概率公式: -0.5 * (dist + log_det + n_features * log(2π))
        const_term = self.n_features * torch.log(torch.tensor(2 * torch.pi))
        log_prob = -0.5 * (dist + log_det.unsqueeze(0) + const_term)
        return log_prob
    
def _custom_distance(self, X):
    # L2距离（欧氏距离平方）
    diff = X.unsqueeze(1) - self.mus.unsqueeze(0)
    dist = (diff ** 2).sum(dim=-1)  # (batch, n_components)
    return dist


def train_gmm(model, data_loader, n_epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        for X_batch in data_loader:
            # logger.debug(f"{X_batch=}")
            optimizer.zero_grad()
            loss = model(X_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(data_loader):.4f}")


from sklearn.cluster import KMeans

def initialize_centers(X, n_components):
    kmeans = KMeans(n_clusters=n_components, n_init=10)
    kmeans.fit(X.numpy())
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)



# 数据生成
torch.manual_seed(42)
X = torch.cat([
    torch.randn(500, 2) * 0.5 + torch.tensor([3.0, 0.0]),
    torch.randn(500, 2) * 0.8 + torch.tensor([-1.0, 2.0]),
    torch.randn(500, 2) * 1.0 + torch.tensor([-3.0, -2.0])
])

# 数据加载器
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(X)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 模型初始化
model = CustomGMM(n_components=3, n_features=2, covariance_type='diag')
model.mus.data.copy_(initialize_centers(X, 3))  # K-means初始化

# 训练
train_gmm(model, data_loader, n_epochs=50, lr=0.1)

# 预测聚类

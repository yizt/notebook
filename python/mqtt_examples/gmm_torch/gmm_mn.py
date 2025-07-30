import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as dist

# 生成模拟数据（3个高斯分布的混合）
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = []
    # 第一个分布
    data.append(np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//3))
    # 第二个分布
    data.append(np.random.multivariate_normal([4, 4], [[1, -0.5], [-0.5, 1]], n_samples//3))
    # 第三个分布
    data.append(np.random.multivariate_normal([-4, 4], [[1, 0.5], [0.5, 1]], n_samples//3))
    return np.vstack(data)

# 定义GMM模型
class GMM(nn.Module):
    def __init__(self, n_components=3, n_features=2):
        super().__init__()
        self.n_components = n_components
        
        # 初始化可训练参数
        self.means = nn.Parameter(torch.Tensor(n_components, n_features))
        self.covars = nn.Parameter(torch.eye(n_features).repeat(n_components, 1, 1))
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        
        # 确保协方差矩阵正定
        self.covars.data = torch.matmul(self.covars, self.covars.transpose(-1, -2)) + torch.eye(n_features).unsqueeze(0)
        
    def forward(self, x):
        # 计算每个分布的log概率
        log_probs = []
        for k in range(self.n_components):
            mvn = dist.MultivariateNormal(self.means[k], self.covars[k])
            log_probs.append(mvn.log_prob(x).unsqueeze(1))
        
        # 合并所有分布的log概率
        log_probs = torch.cat(log_probs, dim=1)
        
        # 计算加权log概率（加上log权重）
        weighted_log_probs = log_probs + torch.log(self.weights.unsqueeze(0))
        
        # 计算log似然（取log-sum-exp）
        log_likelihood = torch.logsumexp(weighted_log_probs, dim=1)
        return -log_likelihood.mean()  # 返回平均负对数似然

# 训练配置
n_components = 3
n_features = 2
n_epochs = 100
learning_rate = 0.01

# 生成数据
data = generate_data()
x = torch.tensor(data, dtype=torch.float32)

# 初始化模型
gmm = GMM(n_components, n_features)
optimizer = optim.Adam(gmm.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = gmm(x)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 可视化结果
def plot_gmm_(gmm, data):
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5, label='Data')
    
    # 绘制高斯分布椭圆
    for k in range(gmm.n_components):
        mean = gmm.means[k].detach().numpy()
        cov = gmm.covars[k].detach().numpy()
        
        # 绘制协方差椭圆
        U, s, Vt = np.linalg.svd(cov)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(5.991 * s)  # 95%置信椭圆
        
        for nsig in [1, 2, 3]:
            ax = plt.gca()
            ellipse = plt.matplotlib.patches.Ellipse(
                mean, nsig*width, nsig*height, angle, 
                facecolor='none', edgecolor='r', linewidth=1.5)
            ax.add_patch(ellipse)
    
    plt.title('GMM Training Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def plot_gmm(gmm, data):
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5, label='Data')
    
    # 绘制高斯分布椭圆
    for k in range(gmm.n_components):
        mean = gmm.means[k].detach().numpy()
        cov = gmm.covars[k].detach().numpy()
        
        # 计算协方差椭圆的参数
        U, s, Vt = np.linalg.svd(cov)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(5.991 * s)  # 95%置信椭圆
        
        # 绘制不同标准差级别的椭圆
        for nsig in [1, 2, 3]:
            ax = plt.gca()
            # 修正：将中心坐标作为元组传递
            ellipse = plt.matplotlib.patches.Ellipse(
                (mean[0], mean[1]),  # 中心坐标作为元组
                nsig*width,          # 宽度
                nsig*height,         # 高度
                angle,               # 旋转角度（度）
                facecolor='none',
                edgecolor='r',
                linewidth=1.5,
                label=f'Component {k+1}' if k==0 else None  # 只添加一次图例
            )
            ax.add_patch(ellipse)
    
    plt.title('GMM Training Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

plot_gmm(gmm, data)
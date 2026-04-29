import numpy as np

class LogisticRegressionFromScratch:
    """
    基于纯 NumPy 实现的逻辑回归分类器。
    
    参数:
    learning_rate (float): 梯度下降的学习率
    num_iterations (int): 训练迭代次数
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Sigmoid 激活函数。
        使用 np.clip 限制 z 的范围，防止 np.exp(-z) 计算时发生溢出 (Overflow)。
        """
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        拟合模型（训练阶段）。
        
        参数:
        X (numpy.ndarray): 训练特征，形状为 (m_samples, n_features)
        y (numpy.ndarray): 训练标签，形状为 (m_samples,)
        """
        num_samples, num_features = X.shape
        
        # 1. 初始化权重和偏置为 0
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 2. 梯度下降主循环
        for _ in range(self.num_iterations):
            # 前向传播：计算线性组合和预测概率 (Hypothesis)
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # 反向传播：计算梯度 (Derivatives)
            # dw = (1/m) * X^T * (h - y)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/m) * sum(h - y)
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # 3. 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        返回样本属于正类 (label=1) 的概率。
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        根据指定的阈值进行二分类预测。
        """
        probabilities = self.predict_proba(X)
        # 将概率大于阈值的转换为 1，否则为 0
        y_predicted_cls = [1 if p > threshold else 0 for p in probabilities]
        return np.array(y_predicted_cls)
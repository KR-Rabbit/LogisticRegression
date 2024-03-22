<h1 align="center">逻辑回归😄</h1>

### 项目结构

```
├── data
│   ├── MNIST 手写数字数据集
│   ├── data/breast_cancer_wisconsin.xlsx 乳腺癌数据集
│   └── README.md 数据集说明
├── util 
│   ├── __init.py__ 定义了MINIST数据集的加载函数
│   └── model.py 定义了模型的训练、预测、评估函数以及正则化函数
├── LogisticRegression.ipynb 逻辑回归模型
├── MultiLogisticRegression.ipynb 多分类逻辑回归模型
└── Newton.ipynb 牛顿法求解逻辑回归模型
```

### 正则化

- L1正则化<br>
  L1正则化的多元逻辑回归模型的损失函数为：

$$
J(θ）= \frac{1}{m}\sum_{i=1}^m[y^{(i)}log\{3h_{\theta}(x^{(i)}\}+(1-y^{(i)})log\{1-h_\theta(x^{(i)}\}]
+\frac{\lambda}{2m}\sum_{j=1}^n|\theta_j|
$$

导数为：

$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m[h_\theta(x^{(i)})-y^{(i)}]x_j^{(i)
}+\frac{\lambda}{m}sign(\theta_j)
$$

L2正则化
L2正则化的多元逻辑回归模型的损失函数为：

$$
J(\theta)=-\frac1m\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)}log\frac{e^{\theta_k^Tx^{(i)}}}{\sum_{j=1}^Ke^{\theta_j^Tx^{(i)
}}}+\frac\lambda{2m}\sum_{j=1}^K\sum_{k=0}^n\theta_{jk}^2
$$

导数为：

$$
\begin{aligned}\frac{\partial J}{\partial\theta_j}&=\frac{1}{m}\sum_{i=1}^m\Big(h_\theta(x^{(i)})-y^{(i)}\Big)x_j^{(i)
}+\frac{\lambda}{m}\theta_j\end{aligned}
$$

### 牛顿法

牛顿法函数公式为

$$
x_{n+1}=x_n-\frac{f^{\prime}(x_n)}{f^{\prime\prime}(x_n)}
$$

二阶导数，即Hessian矩阵为：

$$
H=\frac{\partial^2J(\theta)}{\partial\theta\partial\theta^T}=\sum_{i=1}^mh_\theta(x^{(i)})(1-h_\theta(x^{(i)}))x^{(i)}(
x^{(i)})^T
$$

矩阵形式为：

$$
\theta^{(t+1)}=\theta^{(t)}-H^{-1}\nabla_\theta J(\theta)
$$

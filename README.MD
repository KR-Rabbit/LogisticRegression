### 项目结构

├── **data** 数据文件夹  
│   ├── MNIST 手写数字数据集  
│   ├── data/breast_cancer_wisconsin.xlsx 乳腺癌数据集  
│   └── README.md 数据集说明  
├── **util**  
│   ├── _\_init.py__ 定义了MINIST数据集的加载函数  
│   └── model.py 定义了模型的训练、预测、评估函数以及正则化函数  
├── [**LogisticRegression.ipynb**](LogisticRegression.ipynb) 逻辑回归模型  
├── [**MultiLogisticRegression.ipynb**](MultiLogisticRegression.ipynb) 多分类逻辑回归模型  
└── [**Newton.ipynb**](Newton.ipynb) 牛顿法求解逻辑回归模型
//可折叠的目录树

### 正则化
- L1正则化<br>
L1正则化的多元逻辑回归模型的损失函数为：
![L1.png](image%2FL1.png)
导数为：
![L1gradient.png](image%2FL1gradient.png)
L2正则化<br>
L2正则化的多元逻辑回归模型的损失函数为：
![L2.png](image%2FL2.png)
导数为：
![L2gradient.png](image%2FL2gradient.png)<img src="image/L2gradient.png" alt="">

### 牛顿法
牛顿法函数公式为
![newton.png](image%2Fnewton.png)
二阶导数，即Hessian矩阵为：
![Hessian.png](image%2FHessian.png)
矩阵形式为：
![newton_matrix.png](image%2Fnewton_matrix.png)


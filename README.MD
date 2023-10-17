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
<img src="image/L1.png" alt="">
导数为：
<img src="image/L1gradient.png" alt="">
L2正则化<br>
L2正则化的多元逻辑回归模型的损失函数为：
<img src="image/L2.png" alt="">
导数为：
<img src="image/L2gradient.png" alt="">

### 牛顿法
牛顿法函数公式为
<img src="image/newton.png" alt="">
二阶导数，即Hessian矩阵为：
<img src="image/Hessian.png" alt="">
矩阵形式为：
<img src="image/newton_matrix.png" alt="">


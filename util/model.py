import numpy as np
from tqdm import tqdm


class LogisticRegression:
    def __init__(self, learning_rate=1e-2, epochs=200, regularization=None):
        self.__learning_rate = learning_rate
        self.__epochs = int(epochs)
        self.__regularization = regularization
        self.__weights = None
        self.__loss_list = []
        self.__accuracy_list = []

    @property
    def w(self) -> float:
        return self.__weights[1].item()

    @property
    def b(self) -> float:
        return self.__weights[0].item()

    @property  # for debug
    def weight(self):
        return self.__weights

    @property
    def loss(self):
        """
        训练过程中的loss
        """
        return self.__loss_list

    @property
    def accuracy(self):
        """
        训练过程中的accuracy
        """
        return self.__accuracy_list

    def cal_gradient(self, x, y, y_pred):
        # 偏导数,包括正则化项的偏导数
        # dJ/dt =  1/m * sum((y-y_pred)*x)
        gradient_weight = np.dot(x.T, y_pred - y) / len(x)
        gradient_regularization = self.__regularization.gradient(self.__weights) if self.__regularization else 0
        return gradient_weight + gradient_regularization

    # 交叉熵损失函数
    def cal_loss(self, y_true, y_pred):
        # J = -1/m * sum(y*log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        regular_loss = self.__regularization(self.__weights) if self.__regularization else 0
        return loss + regular_loss

    @staticmethod
    def softmax(data):
        return np.exp(data) / np.sum(np.exp(data), axis=1, keepdims=True)  # keepdims=True 保持维度

    @staticmethod
    def get_batch(data_x, data_y, batch_size=64):
        assert len(data_y) == len(data_x)
        data_index = [np.random.randint(0, len(data_y) - 1) for i in range(batch_size)]
        return data_x[data_index], data_y[data_index]

    @staticmethod
    def one_hot_encode(y):
        # y独热编码
        y_one_hot = np.zeros((len(y), len(np.unique(y))))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot

    @staticmethod
    def one_hot_decode(y):
        # y解码，返回最大值的索引
        return np.argmax(y, axis=1)

    def record(self, l_, acc_):
        # 记录loss和accuracy
        self.__loss_list.append(l_)
        self.__accuracy_list.append(acc_)

    # 逻辑回归，多分类
    def fit(self, x: np.ndarray, y: np.ndarray, method="SGD"):
        y = self.one_hot_encode(y)
        # 初始化权重
        self.__weights = np.random.randn(x.shape[1], y.shape[1])
        bar = tqdm(range(self.__epochs), position=0, ncols=80)

        # 随机梯度下降
        if method.upper() == "SGD":
            for epoch in bar:
                bar.set_description(f"[epoch:{epoch + 1}/{self.__epochs}]")
                idx = np.random.randint(x.shape[0])  # 随机取值 x_
                x_ = x[[idx]]  # 取得idx行,等价于x_[[idx]], x_[[idx]]会减少一个维度
                y_ = y[[idx]]
                y_pred = self.softmax(x_.dot(self.__weights))
                gradient = self.cal_gradient(x_, y_, y_pred)
                self.__weights -= self.__learning_rate * gradient
                loss = self.cal_loss(y_, y_pred)  # 实数
                acc = self.cal_accuracy(y_, y_pred)
                self.record(loss, acc)
                bar.set_postfix(loss=loss, acc=acc)

        # 随机小批量梯度下降
        elif method.lower() == "mini-batch":
            for epoch in bar:
                bar.set_description(f"[epoch:{epoch + 1}/{self.__epochs}]")
                loss_record = []
                acc_record = []
                for batch in range(x.shape[0] // 64):
                    x_, y_ = self.get_batch(x, y, 64)
                    y_pred = self.softmax(x_.dot(self.__weights))
                    gradient = self.cal_gradient(x_, y_, y_pred)
                    self.__weights -= self.__learning_rate * gradient
                    loss = self.cal_loss(y_, y_pred)
                    acc = self.cal_accuracy(self.one_hot_decode(y_), self.one_hot_decode(y_pred))
                    loss_record.append(loss)
                    acc_record.append(acc)

                mean_loss = np.mean(loss_record)
                mean_acc = np.mean(acc_record)
                self.record(mean_loss, mean_acc)
                bar.set_postfix(loss=mean_loss, acc=mean_acc)
        else:
            # 批量梯度下降
            for epoch in bar:
                bar.set_description(f"[epoch:{epoch + 1}/{self.__epochs}]")
                # 前向传播
                y_pred = self.softmax(x.dot(self.__weights))
                # 反向传播
                gradient = self.cal_gradient(x, y, y_pred)
                # 更新权重
                self.__weights -= self.__learning_rate * gradient
                # 记录
                loss = self.cal_loss(y, y_pred).item()
                # 计算精度
                acc = self.cal_accuracy(self.one_hot_decode(y), self.one_hot_decode(y_pred))
                self.record(loss, acc)
                bar.set_postfix(loss=loss, acc=acc)
        bar.close()

    def predict(self, x: np.ndarray):
        # 预测
        y_pred = self.softmax(x.dot(self.__weights))
        return np.argmax(y_pred, axis=1)

    @staticmethod
    def cal_accuracy(y_true, y_pred):
        # 计算精度
        return np.sum(y_true == y_pred) / len(y_true)


class Regular:
    def __init__(self, lamda=0.5, method="L2"):
        self.__lamda = lamda
        self.__method = method

    def __call__(self, weights) -> float:
        if self.__method.upper() == "L2":
            return self.__lamda * np.sum(weights) / len(weights)
        elif self.__method.upper() == "L1":
            return self.__lamda * np.sum(np.abs(weights)) / len(weights)
        else:
            return 0.0

    def gradient(self, weights):
        # 正则化项的梯度
        if self.__method.upper() == "L2":
            return self.__lamda * weights
        elif self.__method.upper() == "L1":
            return self.__lamda * np.sign(weights)
        else:
            return 0

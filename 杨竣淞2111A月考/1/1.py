# 1.
# 使用深度学习框架（tensorflow2
# .0 + keras或pytorch）完成VGG16模型对飞机车鸟数据集处理
# 数据处理：
# (1)
# 声明函数实现根据路径读取图像矩阵，并对图像像素值进行归一化处理
import os

import matplotlib.pyplot as plt

img=r'"D:\GPT浏览器下载\07-08-1-00002\07-08-1-00002\data3"'
# (2)
# 对os.listdir
# 的返回结果遍历读取并处理
def imgs(s):
    return plt.imread(s)/255.0
x,y=[],[]
for i in os.listdir:
    x.append()
    y.append(len(i[0]))
# (3)
# 构建图像矩阵列表及标签列表
# (4)
# 切分训练集和测试集
from sklearn.model_selection import train_test_split
x1,x2,xy,y2=train_test_split(x,y,train_size=0.7)
# (5)
# 改变合适的维度
x1 = x1.reshape(-1, 28, 28, 3) / 255.0
x2 = x2.reshape(-1, 28, 28, 3) / 255.0

# 模型部分：
#
# (6)
# 依据上图红框部分构建VGG网络模块，通过类封装声明vgg16模型，类命名为VGG16
from tensorflow.keras import models,layers, losses, optimizers, activations
class VGG16(models.Model):
    def __init__(self):
        super(self,VGG16).__init__()
        self.s=models.Sequential([
            layers.Conv2D(16,(5,5),padding='same',activation=activations.relu),
            layers.Conv2D(16,(5,5),padding='same',activation=activations.relu),
            layers.MaxPooling2D((2,2)(2,2)),
            layers.Conv2D((16*2),(5,5),padding='same',activation=activations.relu),
            layers.Conv2D((16*2),(5,5),padding='same',activation=activations.relu),
            layers.MaxPooling2D((2,2)(2,2)),
            layers.Conv2D((16*4),(5,5),padding='same',activation=activations.relu),
            layers.Conv2D((16*4),(5,5),padding='same',activation=activations.relu),
            layers.Conv2D((16*4),(5,5),padding='same',activation=activations.relu),
            layers.MaxPooling2D((2,2)(2,2)),

            layers.Flatten(),

            layers.Dense(),
            layers.Dropout(),
            layers.Dense(),
            layers.Dropout(),
            layers.Dense(),
            layers.Dropout(),
        ])
    def call(self, inputs, training=None, mask=None):
        return self.s(inputs)
# (7)
# 类中封装功能层容器
# Sequential
# (8)
# 将图中卷积模块的通道数减半处理
# (9)
# 设定适当的全连接神经元个数
# (10)
# 设定适当的输出层维度数
# (11)
# 设置正向传播方法
# (12)
# 打印VGG16网络参数列表
model=VGG16()
model.build(input_shape=(None,32,32,3))
model.summary()
model.compile(
    optimizer=optimizers.RMSprop(),
    loss=losses.sparse_categorical_crossentropy,
    metrics='acc'
)
# (13)
# 使用训练集对模型拟合训练，参数自拟
# (14)
# 利用测试集进行模型评估
mp=model.predict(x2)
model(y2,mp)
#
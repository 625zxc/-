# 2.
# 卷积神经网络常用模型有：使用leNet5模型，进行mnist数据集训练和分类。利用tensorflow2
# .0
# 或者pytorch深度学习平台，按照下述要求，完成代码编程和演示
# (1)
# 导入数据集
from tensorflow.python.keras.datasets import mnist
from tensorflow.keras import utils,layers,models,activations,losses,optimizers

(x1,y1),(x2,y2)=mnist.load_data()
# (2)
# 浮点数归一化处理
x1=x1.reshape(-1,28,28,1)/255.0
x2=x2.reshape(-1,28,28,1)/255.0

# (3)
# 对数据做独热处理
# (4)
# 根据结构图类封装或者Sequential创建LeNet模型


class leNet5(models.Model):
    def __init__(self):
        super(self, leNet5).__init__()
        self.s = models.Sequential([
            layers.Conv2D(),
            layers.Conv2D(),
            layers.MaxPooling2D(),
            layers.Conv2D(),
            layers.Conv2D(),
            layers.MaxPooling2D(),

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

#
# (5)
# 选择适当的激活函数
# (6)，
# 配置模型优化器、损失函数、评估函数
model=leNet5()
model.build(input_shape=(None,32,32,1))
model.summary()
model.compile(
    optimizer=optimizers.RMSprop(),
    loss=losses.sparse_categorical_crossentropy,
    metrics='acc'
)
# (7)
# 训练模型：批量数据100，迭代10次
model.fit(x1,y1,batch_size=100,epochs=10)
# (8)
# 利用测试集进行模型评估
mp=model.predict(x2)
model(y2,mp)
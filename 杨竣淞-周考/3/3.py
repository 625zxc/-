#
# 2.	按照要求，完成VGG16以下处理
# ①	数据处理
# 1)	读取mnist数据集
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers,losses,activations,optimizers,metrics,models
(train_x,train_y),(tast_x,tast_y)=mnist.load_data()
# 2)	对数据进行维度转换、归一化等相关预处理
train_x=train_x.reshape(-1,28,28,1).astype('float32')/255.0
tast_x=tast_x.reshape(-1,28,28,1).astype('float32')/255.0
# ②	设置VGG16模块（类），
# 1)	声明一个Sequential包括网络结构中的所有功能层
class VGG16(models.Model()):
    def __init__(self):
        super(VGG16,self).__init__()
        self.seq=Sequential([
            layers.Conv2D(16,(3,3),padding='same',activation=activations.relu),
            layers.Conv2D(16, (3, 3), padding='same', activation=activations.relu),
            layers.Conv2D(16, (3, 3), padding='same', activation=activations.relu),
            layers.MaxPooling2D((2,2),(2,2)),
            layers.Conv2D(16*2, (3, 3), padding='same', activation=activations.relu),
            layers.Conv2D(16*2, (3, 3), padding='same', activation=activations.relu),
            layers.Conv2D(16*2, (3, 3), padding='same', activation=activations.relu),
            layers.MaxPooling2D((2,2),(2,2)),
            layers.Conv2D(16*2*2, (3, 3), padding='same', activation=activations.relu),
            layers.Conv2D(16*2*2, (3, 3), padding='same', activation=activations.relu),
            layers.Conv2D(16*2*2, (3, 3), padding='same', activation=activations.relu),
            layers.MaxPooling2D((2, 2), (2, 2)),
            layers.Conv2D(16*2*2*2, (3, 3), padding='same', activation=activations.relu),
            layers.Conv2D(16*2*2*2, (3, 3), padding='same', activation=activations.relu),
            layers.Conv2D(16*2*2*2, (3, 3), padding='same', activation=activations.relu),
            layers.MaxPooling2D((2,2),(2,2)),

            # 2)	根据下图VGG16网络结构构建模型类
            # 3)	卷积模型取前四组，且初始卷积核个数为16
            # 4)	每到一个新的卷积组通道数翻倍
            # 5)	最后三层全链接通道数分别为2048，512，10
            # 6)	Dropout层失活率设置为0.4
            layers.Flatten(),

            layers.Dense(2048, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(512,activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(10,activation='softmax')
        ])
    def __call__(self, *args, **kwargs):
        return self.seq(*args)
# 7)	实现正向传播处理
m=models.Model(input_shape=(None,32,32,1))
# ③	完成模型创建及训练,

m.compile(
    optimizer=optimizers.Adam(),
    loss=losses.sparse_categorical_crossentropy,
    metrics='mas'
)
m.fit(train_x,train_y)
# 1)	输出模型检验后最终的损失值，准确率
loss,acc=m.history('loss'),m.history('acc')
print(loss,acc)
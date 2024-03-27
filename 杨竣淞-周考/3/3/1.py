# 1.	按照要求使用keras，搭建rnn处理以下内容 sample = "hihello"
# (1)	数据预处理
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers,losses,optimizers,activations,models,metrics

sample = "hihello"
# ①	将出现的单词按照字典形式进行处理
# ②	使用上面的sample，将hihell作为特征，ihello作为标签
# ③	设置合理的时间序列，将x进行对应的处理
# (2)	模型操作
# ①	使用LSTM模型进行处理
class LSTM(models.Model()):
    def __init__(self):
        super(LSTM,self).__init__()
        self.seq=Sequential([
            layers.Conv2D(16,(3,3),padding='same',activation=activations.relu),
            layers.Conv2D(16, (3, 3), padding='same', activation=activations.relu),
            layers.MaxPooling2D((2,2),(2,2)),

            layers.Flatten(),

            layers.Dense(2048, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(512,activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(10,activation='softmax')
        ])
    def __call__(self, *args, **kwargs):
        return self.seq(*args)
# ②	叠加一层lstm模型，元素数量相同
# ③	使用对应方式将数据进行softmax处理
# ④	合理编译模型
# ⑤	训练模型
m=models.Model(input_shape=(None,32,32,1))

m.compile(
    optimizer=optimizers.Adam(),
    loss=losses.sparse_categorical_crossentropy,
    metrics='mas'
)
m.fit(train_x,train_y)
m.predict(test_x)
loss,acc=m.history('loss'),m.history('acc')
print(loss,acc)
# ⑥	预测结果
# ⑦	将预测结果进行打印，核对结果


import numpy as np
from tensorflow.keras import utils,layers,models,activations,losses,optimizers


# 3． 用tensorflow2.X或者Pytorch实现循环神经网络文本预测
问="do you love me?"
答="yes,me too love"
# （1）将数据中转换成字典类型进行处理
print(问,答)
char_set=list(set(问+答))
print(char_set)
# （2）将问句作为输入值，答句作为预测值
# （3）对应已构建好的字典对输入句和预测句进行编码
char2id = {j: i for i, j in enumerate(char_set)}
id2char = {i: j for i, j in enumerate(char_set)}
x,y=[],[]
# （4）构建句长和词向量长
# （5）将已经编码好的句子进行独热处理

from sklearn.model_selection import train_test_split
x1,x2,xy,y2=train_test_split(x,y,train_size=0.7)
x1=x1.reshape(-1,28,28,1)/255.0
x2=x2.reshape(-1,28,28,1)/255.0
y1=utils.to_categorical(10)
y2=utils.to_categorical(10)
# （6）再进行特征维度调整，调整为3维矩阵
# （7）用类封装或Sequential定义两层LSTM模型进行处理，两层设定相同的输出单元数
class LSTM(models.Model):
        def __init__(self):
            super(self, LSTM).__init__()
            self.s = models.Sequential([
                layers.LSTM(),
                layers.LSTM(),
                layers.Dropout()
            ])
        def call(self, inputs, training=None, mask=None):
            return self.s(inputs)
# （8）接一层全连接作为输出层，并选择合适的激活函数
# （9）模型编译，配合合理的优化器，损失函数及评价指标
model=LSTM()
model.build(input_shape=(None,32,32,1))
model.summary()
model.compile(
    optimizer=optimizers.Adam(0.01),
    loss=losses.sparse_categorical_crossentropy,
    metrics='acc'
)
# (7)
# 训练模型：批量数据100，迭代10次
model.fit(x1,y1,batch_size=100,epochs=10)
# (8)
# 利用测试集进行模型评估
pred=model.predict(x)
for i,j in enumerate(pred):
    pred_index=np.argmax(j,axis=1)
    pred_value=[id2char[c] for c in pred_index]
    print('pred_value',''.join(pred_value))
# （10）得到预测的结果

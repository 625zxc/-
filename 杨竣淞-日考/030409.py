# 照要求，使用rnn处理以下内容（每题10分）		sample = "hihello"
# (1)	数据预处理
# ①	将出现的单词按照字典形式进行处理
from tensorflow.keras import layers,models,utils
sample = "hihello"
slist=list(sample)
hihell={j:i for i,j in enumerate(slist)}
ihello={i:j for i,j in enumerate(slist)}
hl=len(hihell)
xs=sample[:-1]
ys=sample[1:]
x2=[hihell[i] for i in xs]
y2=[hihell[i] for i in ys]
xl=len(x2)
lslist=len(slist)
x=utils.to_categorical(x2,lslist).reshape(-1,xl,lslist)
y=utils.to_categorical(y2,lslist).reshape(-1,xl,lslist)

# ②	使用上面的sample，将hihell作为特征，ihello作为标签
# ③	设置合理的时间序列，将x进行对应的处理
# (2)	模型操作
# ①	使用LSTM模型进行处理
m=models.Sequential([
    layers.LSTM(lslist, input_shape=(xl,lslist), return_sequences=True),
    layers.LSTM(lslist, return_sequences=True),
    layers.TimeDistributed(layers.Dense(lslist,activation='softmax')),

])
# ②	叠加一层lstm模型，元素数量相同
# ③	使用对应方式将数据进行softmax处理
# ④	合理编译模型
m.compile(optimizer="Adam",
          loss='categorical_crossentropy',
          metrics='acc')
# ⑤	训练模型
# ⑥	预测结果
mfit=m.fit(x,y,epochs=100)
# ⑦	将预测结果进行打印，核对结果
for pc in m.predict_classes(x):
    yc=[ihello[i] for i in pc]
    print(''.join(yc))
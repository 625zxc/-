#
# ①　导入TensorFlow库
# ②　导入numpy
# ③　从TensorFlow中的keras模块中导入    models, Sequential, losses, optimizers, layers
# ④　从'bodyfat.csv'文件中加载数据到数组data中
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers,Sequential,losses,activations,optimizers,models
data=np.loadtxt('bodyfat.csv',delimiter=',')
# ⑤　获取data数组的所有行，但从第一列开始的所有列，赋值给x_data
y_data=data[:,0]
x_data=data[:,:1]
# ⑥　获取data数组的所有行，第0列的数据，赋值给y_data
# ⑦　从sklearn库中的preprocessing模块导入StandardScaler类
# x_data=StandardScaler.fit_transform(x_data)
# ⑧　使用StandardScaler对x_data进行数据标准化处理
# ⑨　从sklearn库中的model_selection模块导入train_test_split函数
x1,x2,y1,y2=train_test_split(x_data,y_data)
# ⑩　使用train_test_split对x_data和y_data进行数据集划分
# ⑪　在Sequential模型中新增一个具有128个神经元并使用ReLU激活函数的全连接层
m=Sequential([
    layers.Dense(128,activation=activations.relu),
    layers.Dense(256, activation=activations.relu),
    layers.Dense(1)
])
# ⑫　在Sequential模型中新增一个具有256个神经元并使用ReLU激活函数的全连接层
# ⑬　在Sequential模型中新增一个具有1个神经元的全连接层
# ⑭　编译模型，使用Adam优化器，mse作为损失函数，mse作为评估指标
m.compile(optimizer=optimizers.Adam(),
          loss=losses.mse,
          metrics='mse')
# ⑮　使用训练集训练模型，进行1000轮训练，每批次大小为30，并将训练过程记录到log中
log=m.fit(x1,y1,epochs=1000,batch_size=30)
# ⑯　打印模型结构
m.summary()
# ⑰　打印训练过程中记录的损失值
print(log.history['loss'])
print(log.history['mse'])
# ⑱　打印训练过程中记录的均方误差值
# ⑲　对测试集进行预测
print(m.predict(x2))

# ⑳　对测试集进行模型评估
print(m.evaluate(x2,y2))
# 21　保存模型到'bodyfat.h5'文件
m.save('bodyfat.h5')
# 22　从'bodyfat.h5'文件中加载模型到save_model
save_model=models.load_model('bodyfat.h5')
# 23　使用测试集数据对加载的模型进行训练，进行100轮训练，每批次大小为30
save_model.fit(x2,y2,epochs=1000,batch_size=30)
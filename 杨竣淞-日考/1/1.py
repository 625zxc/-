# 1.	对以下数据进行多变量线性回归处理（每小题10分）
# (1)	数据处理
# ①	读取data-01-test-score.csv数据
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.models

data=np.loadtxt('data-01-test-score.csv',delimiter=',')
# ②	将最后一列作为y标签，其他数据作为x
x=data[:,:-1]
y=data[:,-1:]
# ③	将数据进行洗牌处理
# ④	将数据按照7:3比例切分为训练集和测试集
from sklearn.model_selection import train_test_split
x1,x2,y1,y2=train_test_split(x,y,test_size=0.3,shuffle=True)
# (2)	模型操作
# ①	创建模型
m=tensorflow.keras.models.Sequential(
    tensorflow.keras.layers.Dense(7)
)
# ②	设置网络，输入3个特征，输出一个数据值
# ③	使用rmsprop进行梯度下降，损失函数设置为均方误差
m.compile(optimizer='RMSprop',loss='mse')
mfit=m.fit(x1,y1,epochs=2000)
# ④	训练数据2000次
# ⑤	使用测试集数据，打印测试结果
print(m.evaluate(x2,y2))
# ⑥	可视化：将测试集的真实值和预测值变化绘图进行对比
plt.plot(m.predict(x2),'y')
plt.plot(y2,'r')
plt.show()
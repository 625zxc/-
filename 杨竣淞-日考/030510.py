# 人工智能学院《深度学习二》日考-技能
#
# 题号	一	二	总分	批卷人	审核人
# 得分
#
# 一、技能考试时间为：  1  小时
# 二、技能题（共100分）
# （一）题目要求：
# 1.	按要求完成	下面的各项需求。
# 2.	必须有录屏，无录屏者一律0分处理，必须是完整的考试录屏(只有单独录效果录屏按0分处理)，录屏过程中不允许有暂停行为，若是发现，按考试作弊处理，桌面必须有自己的学院、班级、姓名。
# 3.	上交U盘时，U盘中只允许有自己考试的项目，否则按零分处理。
# （二）评分要求：
# 1.	按照要求，使用rnn完成股票预测（10分）
# (1)	数据处理
# ①	读取数据，进行倒序处理
# ②	数据进行归一化处理
import matplotlib.pyplot as plt
import numpy as np

data=np.loadtxt(r'C:\Users\DELL\PycharmProjects\pythonProject57\030510\data-02-stock_daily.csv')
from sklearn.preprocessing import MinMaxScaler
data=MinMaxScaler.fit_transform(data)
xy=data[::-1]
datax,datay=[],[]
nl=7
for i in range(0,len(xy)+nl):

    datay.append({i: i + nl})
    datax.append({i: i})
# ③	将数据全部列作为x，最后一列作为y
# ④	设置数据前7天作为x，第8天作为y
datax=np.array(datax)
datay=np.array(datay)

# ⑤	将数据按照7:3切分
from sklearn.model_selection import train_test_split
x1,x2,y1,y2=train_test_split(datax,datay,train_size=0.7)
# (2)	模型处理
# ①	创建模型
from tensorflow.keras import models,losses,layers
# ②	使用lstm进行处理
m=models.Sequential([
    layers.LSTM(nl,input_shape=(xy.shape[1],nl),activation='softmax')
])
# ③	编译模型，使用mse，配合adam优化器
m.compile(optimizer='Adam',
          loss=losses.mse,
          metrics='mse')
# ④	预测结果
mfit=m.fit(x1,y1,epochs=100)
pm=m.predict(x2)
# ⑤	将实际值和预测值结果可视化
plt.plot(datax)
plt.plot(pm)
plt.show()
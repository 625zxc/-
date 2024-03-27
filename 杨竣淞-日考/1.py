# 1.	使用tensorflow2.0完成以下操作（每小题10分）
# (1)	矩阵创建
# ①	创建一个维度为3*3的全1矩阵
import tensorflow

a = tensorflow.ones((3, 3))
# ②	使用range，创建一个1-9的1阶张量
b = tensorflow.range(1, 9)
print(b)
# ③	打印上题的维度
# ④	将上题维度修改为3,1,3
b1 = b.reshape(-1,shape=(3, 1, 3))
# ⑤	使用函数，去除维度中函数1的维度
b2 = tensorflow.squeeze(b1)
# (2)	切片及其他
# ①	使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
# ②	打印上题向量的均值
b4 = tensorflow.random.normal((2, 2))[2:4]
# ③	创见一个2行2列的标准正态分布矩阵
b5 = tensorflow.zeros((2, 2))
# ④	创建一个2行2列的全0矩阵
# ⑤	将3,4问的结果拼接成一个4行2列的结果
b6 = tensorflow.concat((b4, b5), axis=0)
print(b1, b2, b4, b5, b6)

import tensorflow

a1=tensorflow.ones((3,3))
print(a1)

a2=tensorflow.range(1,10)
print(a2,'\t',a2.shape)

a3=tensorflow.reshape(a2,[3,1,3])
print(a3,'\t',a3.shape)

a4=tensorflow.squeeze(a3)
print('................................',a4,'\t',a4.shape)

b1=tensorflow.range(1,10)[2:6]
print('1111111111111',b1,'\t',b1.shape)

c1=tensorflow.random.normal([2,2])
print('2222222222222',c1,'\t',c1.shape)

c2=tensorflow.zeros([2,2])
print('2222222222222',c2,'\t',c2.shape)

c3=tensorflow.concat([c1,c2],axis=0)
print('33333333333',c3,'\t',c3.shape)








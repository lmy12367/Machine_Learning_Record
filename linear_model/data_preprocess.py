import numpy as np
from sklearn.preprocessing import StandardScaler

lines = np.loadtxt("./data/ml/linear_model/USA_Housing.csv",delimiter=",",dtype="str")
header=lines[0]
lines=lines[1:].astype(float)

print("数据特征",','.join(header[:-1]))
print('数据标签',header[-1])
print('数据条数',len(lines))

ratio=0.8
split=int(len(lines)*ratio)
np.random.seed(0)
lines=np.random.permutation(lines)
train,test=lines[:split],lines[split:]

scaler=StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test,  y_test  = test[:, :-1],  test[:, -1].flatten()
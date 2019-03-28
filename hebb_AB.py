import numpy as np


A=np.array([[-1,1,-1],
   [1,-1,1],
   [1,1,1],
   [1,-1,1],
   [1,-1,1]])


B=np.array([[1,1,-1],
   [1,-1,1],
   [1,1,-1],
   [1,-1,1],
   [1,1,-1]])


a=A.reshape(1,-1)
b=B.reshape(1,-1)
y=np.array([1,-1]).reshape(2,1)
print("Y",y)
X=np.append(arr=a,values=b,axis=0)
print("X",X)


W=np.zeros((1,len(X[0])))
b=0



print("\n\nTrainig the network.....")
xy=X*y

for i in range(len(X)):
    W=W+xy[i]
    b=int(b+y[i])
    
print("Final weights:\n",W)
print("Final bias:\n",b)


import numpy as np
import matplotlib.pyplot as plt


data=np.array([[1,1,1],
   [1,-1,-1],
   [-1,1,-1],
   [-1,-1,-1]])

X=data[:,[0,1]]
y=data[:,[-1]]


def init(f_size,rows):
    w=np.zeros((1,f_size),dtype=int)
    b=0
    return (w,b)

weight,bias=init(len(X[0]),len(X))


w={}
b={}
w['w0']=weight
b['b0']=bias


xy=X*y

for i in range(len(X)):
    
    weight=weight+xy[i]
    w['w'+str(i+1)]=weight
    bias=int(bias+y[i])
    b['b'+str(i+1)]=bias
    

z=np.array([-1,1])
for i in range(len(X)):
    plt.scatter(X[:,[0]],X[:,[1]],c=y)
    w1=int(w['w'+str(i+1)][0][0])
    w2=int(w['w'+str(i+1)][0][1])
    b_=int(b['b'+str(i+1)])
    #formula used is x2=((-w1*x1)-b)/w2
    plt.plot(z,((-w1*z)-b_)/w2,label="descision boundry for w"+str(i+1)+"b"+str(i+1))
    plt.legend()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import itertools



data=np.array([[1,1,1],
   [1,-1,-1],
   [-1,1,-1],
   [-1,-1,-1]])

'''sepearating x(x1,x2) and y from above list'''
X=data[:,[0,1]]
y=data[:,[-1]]

'''initialize weights W(w1,w2) and bias b'''
def init(f_size,rows):
    w=np.zeros((1,f_size),dtype=int)
    b=0
    return (w,b)

weight,bias=init(len(X[0]),len(X))

alpha=int(input("Enter learning rate:"))

''''calculate X*y (x1*y,x2*y)'''
xy=X*y*alpha

max_itr=int(input("Enter max Epochs:"))


z=np.array([-1,1])
for j in range(max_itr):
    for i in range(len(X)):
        y_in=bias+int(np.dot(X[i],weight.T))
        
        if y_in > 0:
            Y=1
        elif y_in <1:
            Y=-1
           
        if Y==int(y[i]):
            continue
        else:
            #update weights
            weight=weight+alpha*X[i]*y[i]
            bias=int(bias+alpha*y[i])
            
            #colors=itertools.cycle(["r","b","g"])
            plt.scatter(X[:,[0]],X[:,[1]],c=y)
            w1=int(weight[0][0])
            w2=int(weight[0][1])
            b_=bias
            #x2=((-w1*x1)-b)/w2
            #used above formula to get x2 from Textbook
            plt.plot(z,((-w1*z)-b_)/w2,label="descision boundry ")
            plt.legend()
            plt.show()
    
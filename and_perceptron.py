import numpy as np
import matplotlib.pyplot as plt

#AND INput
data=np.array([[1,1,1],
   [1,-1,-1],
   [-1,1,-1],
   [-1,-1,-1]])

''' sepearating x(x1,x2) and y inputs'''
X=data[:,[0,1]]
y=data[:,[-1]]

'''  initialize weights W(w1,w2) and bias b '''
def init(f_size,rows):
    w=np.zeros((1,f_size),dtype=int)
    b=0
    return (w,b)

weight,bias=init(len(X[0]),len(X))


alpha=int(input("Enter learning rate:"))

'''' calculate X*y (x1*y,x2*y) '''
xy=X*y*alpha


max_itr=int(input("Enter max itr:"))


print("\nTrainig the network.....\n")
for j in range(max_itr):
    for i in range(len(X)):
        
        #weighted sum
        y_in=bias+int(np.dot(X[i],weight.T))
        
        #Apply activation Function
        if y_in > 0:
            Y=1
        elif y_in <=0:
            Y=-1
           
        #Weight updation only if target not equal to predicted
        if Y==int(y[i]):
            continue
        else:
            weight=weight+alpha*X[i]*y[i]
            bias=int(bias+alpha*y[i])
            
print("\nAfter training \nFinal weight:\n",weight,"\nBias:\n",bias)
    
#plottinG
z=np.array([-1,1])
print(z)
plt.scatter(X[:,[0]],X[:,[1]],c=y)
w1=int(weight[0][0])
w2=int(weight[0][1])
b_=bias
#x2=((-w1*x1)-b)/w2
#formula from textbook
plt.plot(z,((-w1*z)-b_)/w2,label="descision boundry ")
plt.legend()
plt.show()

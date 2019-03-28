
import numpy as np

#Input is manually given like 7 segemnt(15)
data={}
data[0]=np.array([[1,1,1],
                 [1,-1,1],
                 [1,-1,1],
                 [1,-1,1],
                 [1,1,1]])
data[1]=np.array([[-1,1,-1],
                 [1,1,-1],
                 [-1,1,-1],
                 [-1,1,-1],
                 [1,1,1]])
data[2]=np.array([[1,1,1],
                 [-1,-1,1],
                 [1,1,1],
                 [1,-1,-1],
                 [1,1,1]])
data[3]=np.array([[1,1,1],
                 [-1,-1,1],
                 [1,1,1],
                 [-1,-1,1],
                 [1,1,1]])
data[4]=np.array([[1,-1,-1],
                 [1,-1,-1],
                 [1,1,1],
                 [-1,-1,1],
                 [-1,-1,1]])
data[5]=np.array([[1,1,1],
                 [1,-1,-1],
                 [1,1,1],
                 [-1,-1,1],
                 [1,1,1]])
data[6]=np.array([[1,1,1],
                 [1,-1,-1],
                 [1,1,1],
                 [1,-1,1],
                 [1,1,1]])
data[7]=np.array([[1,1,1],
                 [-1,-1,1],
                 [-1,-1,1],
                 [-1,-1,1],
                 [-1,-1,1]])
data[8]=np.array([[1,1,1],
                 [1,-1,1],
                 [1,1,1],
                 [1,-1,1],
                 [1,1,1]])
data[9]=np.array([[1,1,1],
                 [1,-1,1],
                 [1,1,1],
                 [-1,-1,1],
                 [1,1,1]])

y_res=list(data.keys())
X_train=np.zeros((0,data[0].shape[0]*data[0].shape[1]))

for char in list(data.keys()):
    X_train=np.append(arr=X_train,values=data[char].reshape(1,-1),axis=0)
    

print(X_train)
#Target of training pattern 
y_train=np.array([[0,0,0,0],
                 [0,0,0,1],
                 [0,0,1,0],
                 [0,0,1,1],
                 [0,1,0,0],
                 [0,1,0,1],
                 [0,1,1,0],
                 [0,1,1,1],
                 [1,0,0,0],
                 [1,0,0,1]])
y_train=np.where(y_train==0,-1,y_train)
print(y_train)
##
#
# perceptron learning
def perceptron(X,y,alpha,epoch):
    n=len(X_train)
    m=len(X_train[0])
    t_n=len(y_train[0])
    
    W=np.zeros((m,t_n))
    bias=np.zeros((t_n,1))

    for itr in range(epoch):
        for i in range(n):
            Y=[]
            for j in range(t_n):
                y_in=float(bias[j]+np.dot(X[i],W[:,[j]]))
                Y.append(y_in)
                
            Y=np.array(Y).reshape(1,-1)
            Y=np.where(Y>0,1,-1)
            for j in range(t_n):
                if(Y[0][j]==y[i][j]):
                    continue
                else:
                    W[:,[j]]=W[:,[j]]+alpha*y[i][j]*X[[i],:].T
                    bias[j]=bias[j]+alpha*y[i][j]
    return W,bias
    
        

print("\n\nTraining the network....\n")
W,bias=perceptron(X_train,y_train,1,epoch=13)
print("\nFinal weigted and bias matrix after training is:\nWeight=\n",W,"\nBias=\n",bias)



print("\n\nTesting  the network with Non Noisy input....\n")
X_test=X_train
count=0
for i in range(X_test.shape[0]):
    x=X_test[[i],:]
    print("\npattern ",i,"is:\n",x.reshape(5,3))
    x=X_test[[i],:]
    y_pred=bias.T+np.dot(x,W)
    
    Y=np.where(y_pred>0,1,-1)
    print("predited target is :\n",Y)
    j=0
    flag=False
    for j in range(len(y_res)):
        if(Y==y_train[[j],:]).all():
            print("Result:\nThe pattren is ",y_res[j])
            flag=True
            break
    if(not(flag)):
        print("Result:\nUnknown pattern")
    else:
        if(i==j):
            count+=1
print("Accuracy: ",100*count/10, "%")



print("\n\nTesting  the network with  Noisy input....\n")
noisy_data={}
noisy_data[0]=np.array([[1,1,-1],
                 [1,-1,1],
                 [1,-1,1],
                 [1,-1,1],
                 [1,1,1]])
noisy_data[1]=np.array([[-1,1,-1],
                 [-1,1,-1],
                 [-1,1,-1],
                 [-1,1,-1],
                 [1,1,1]])

noisy_data[5]=np.array([[1,1,1],
                 [1,-1,-1],
                 [1,1,1],
                 [-1,-1,1],
                 [1,1,-1]])

noisy_data[9]=np.array([[1,1,1],
                 [1,-1,1],
                 [1,1,1],
                 [-1,-1,1],
                 [-1,1,1]])
    
y_res=list(noisy_data.keys())
X_test=np.zeros((0,noisy_data[0].shape[0]*noisy_data[0].shape[1]))

for char in list(noisy_data.keys()):
    X_test=np.append(arr=X_test,values=data[char].reshape(1,-1),axis=0)
    
count=0
for i in range(X_test.shape[0]):
    x=X_test[[i],:]
    print("\npattern ",i,"is:\n",x.reshape(5,3))
    x=X_test[[i],:]
    y_pred=bias.T+np.dot(x,W)
    
    Y=np.where(y_pred>0,1,-1)
    print("predited target is :\n",Y)
    j=0
    flag=False
    for j in range(len(y_res)):
        if(Y==y_train[[j],:]).all():
            print("Result:\nThe pattren is ",y_res[j])
            flag=True
            break
    if(not(flag)):
        print("Result:\nUnknown pattern")
    else:
        if(i==j):
            count+=1
print("Accuracy: ",100*count/len(X_test), "%")
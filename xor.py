import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])

y=np.array([0,1,1,0]).reshape(len(X),1)

def weighted_sum(x,w):
    y=np.dot(x,w)
    return y

def get_activation(y_in,threshold):
    y_out=y_in.copy()
    y_out[y_in>=threshold]=1
    y_out[y_in<threshold]=0
    return y_out

while True:
    #layer 1
    print("Enter layer 1 data for Z1:")
    w11=np.array([[float(input('enter weight w11:'))],[float(input('enter weight w21:'))]])
    th11=float(input('enter threshold th11:'))
    y_in11=weighted_sum(X,w11)
    z1=get_activation(y_in11,th11)
    
    print("Enter layer 1 data for Z2:")
    w12=np.array([[float(input('enter weight 12:'))],[float(input('enter weight 22:'))]])
    th12=float(input('enter threshold th12:'))
    y_in12=weighted_sum(X,w12)
    z2=get_activation(y_in12,th12)
    
    #layer 2
    x=np.append(arr=z1,values=z2,axis=1)
    print("Enter layer 2 data for Z:")
    w2=np.array([[float(input('enter weight for z1:'))],[float(input('enter weight z2:'))]])
    th2=float(input('enter threshold th:'))
    y_in2=weighted_sum(x,w2)
    z=get_activation(y_in2,th2)
    
    print("input:\n",X)
    print("output:\n",z)
    
    if(y==z).all():
        print('network calculated the correct output of XOR logic')
        break
    else:
        print('network did not learn re-enter the weights and the bias')
       
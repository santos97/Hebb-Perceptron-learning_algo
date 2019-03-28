
import numpy as np


X=np.array([[0],[1]])



y=np.array([1,0]).reshape(len(X),1)



def weighted_sum(x,w):
    y=np.dot(x,w)
    return y

# MCP model
while True:
   
    w=np.array([[float(input('enter weight 1:'))]])
    
    
    th=float(input('enter threshold:'))
    
   
    y_in=weighted_sum(X,w)
    
   
    y_out=np.where(y_in>=th,1,0)
    
   
    print("output:\n",y_out)
    
  
    if(y==y_out).all():
        print('network calculated the correct output of NOT logic')
        break
    else:
        print('network did not learn re-enter the weights and the bias')
    

    

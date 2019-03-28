#Basic noise classification
import numpy as np


X=np.array([[0,-1,1],
   [0,1,-1],
   [0,0,-1],
   [0,0,1],
   [0,0,-1],
   [0,1,0],
   [1,0,1],
   [1,0,-1],
   [1,-1,0],
   [1,0,0],
   [1,1,0],
   [0,-1,0],
   [1,1,1]])


W=np.array([int(input("Enter weights:")) for i in range(3)])

#Train 
for i in range(len(X)):
    x=X[i]
    print("\nInput:",x)
    y_in=np.dot(x,W.T)
    if(y_in<0):
        print("Incorrect")
    elif(y_in>0):
        print("Correct")
    else:
        print("Indefinite")
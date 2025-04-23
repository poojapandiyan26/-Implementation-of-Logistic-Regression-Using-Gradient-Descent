# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary

6.Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:P.Pooja Sri 
RegisterNumber:212224230197
*/

import pandas as pd
import numpy as np
df=pd.read_csv("Placement_Data.csv")
df
df=df.drop('sl_no',axis=1)
df=df.drop('salary',axis=1)
df.head()
df["gender"]=df["gender"].astype("category")
df["ssc_b"]=df["ssc_b"].astype("category")
df["hsc_b"]=df["hsc_b"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df["degree_t"]=df["degree_t"].astype("category")
df["workex"]=df["workex"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["status"]=df["status"].astype("category")
df.dtypes
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
y
theta = np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,Y):
    h=sigmoid(x.dot(theta))
    return -np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))

def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(Y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
![image](https://github.com/user-attachments/assets/3934aa52-1fd5-482b-9eb4-ddc311a3cdba)

![image](https://github.com/user-attachments/assets/6fc330e0-d873-4c3d-be65-95f7352a5ac2)

![image](https://github.com/user-attachments/assets/29a507bd-ac25-4457-87ff-2d2190b99dd3)

![image](https://github.com/user-attachments/assets/a856a9b3-1cfe-450d-aede-c6cf62de71a5)

![image](https://github.com/user-attachments/assets/f064c174-6f9b-4eb5-ac77-477609b6bcb4)

![image](https://github.com/user-attachments/assets/2fc5165a-1bc3-49ac-9393-ef9778b07980)

![image](https://github.com/user-attachments/assets/36a04e6b-bac9-48b6-ae0a-6c54126243cd)








## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


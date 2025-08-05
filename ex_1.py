import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read csv("ex_1.csv")
dataset.head()
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.model selection import train test split
xtr,xte, ytr,yte=train_test_split(x,y,test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtr,ytr)
ypr=model.predict(xte)
print("predicted(ypr)")
print("True(yte)")
plt.scatter(xtr, ytr,color='red')
plt.plot(xtr.model.predict(xtr),color='blue')
plt.title("milleage vs selling price (Training set)")
plt.xlabel("milleage")
plt.ylabel("selling_price")
plt.show()
plt.scatter(xte, ytr,color='red')
plt.plot(xte.model.predict(xte),color='blue')
plt.title("milleage vs selling price (Training set)")
plt.xlabel("milleage")
plt.ylabel("selling_price")
plt.show()

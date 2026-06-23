import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
boston=pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
x,y=boston[['rm']].values,boston['medv'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
linreg=LinearRegression()
linreg.fit(x_train,y_train)
y_pred_lr=linreg.predict(x_test)

sort_idx=x_test.flatten().argsort()
xs_test=x_test[sort_idx]
ys_pred=y_pred_lr[sort_idx]
plt.figure(figsize=(15,8))
plt.scatter(x_test,y_test,color="red",label="Actual")
plt.plot(xs_test,ys_pred,color="black",label="Prediction")
plt.title("Linear Regression - Boston Housing")
plt.xlabel("Rooms")
plt.ylabel("Price")
plt.legend()
plt.show()

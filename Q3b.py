import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read csv file
dataSet = pd.read_csv('data.csv')
x = dataSet.iloc[:,:1]
y = dataSet.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=1/3, random_state=42)


from sklearn.linear_model import LinearRegression
reg1 = LinearRegression()
reg1.fit(x_train, y_train)
y_pred = reg1.predict(x_test)

#scatter and plot
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg1.predict(x_train), color="blue")
plt.title("Train data scatter plot")
plt.xlabel("xlabel:-1")
plt.ylabel("ylabel-A")
plt.show()


#test data plotting
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg1.predict(x_test), color="red")
plt.title("Test data scatter plot")
plt.xlabel("xlabel:-1")
plt.ylabel("ylabel-A")
plt.show()

#Performance Evaluation
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print("Performance Evaluation\n")
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

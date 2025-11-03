import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

data= load_diabetes()
x=data.data
y=data.target

#data=pd.read_csv(r'C:\Users\HP\Downloads\carprice.csv')
#data=data.replace('?', None)
#data=data.dropna()

#x=data[['carlength']]
#y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print('slope:',model.coef_[0])
print('intercept:',model.intercept_)
print('mse:',mean_squared_error(y_test,y_pred))

plt.scatter(x[:,0],y, color='blue',label='Actual Data')
plt.plot(x[:,0],model.predict(x), color='red',label='Regression Line')
plt.xlabel('Length of Car')
plt.ylabel('Price of Car')
plt.title('Car Price vs Length')
plt.legend()
plt.show()



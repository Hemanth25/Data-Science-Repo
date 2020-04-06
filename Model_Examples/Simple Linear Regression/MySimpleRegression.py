import pandas as pd

import matplotlib.pyplot as plt

Data = pd.read_csv('Salary_Data.csv')
X = Data.iloc[:,:-1].values
y = Data.iloc[:,1].values

#Test set and training set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y, test_size=1/3 , random_state=0)


#regression model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

Y_pred = regression.predict(X_test)


#plotting

plt.scatter(X_train, Y_train , color = "Red")
plt.plot(X_train, regression.predict(X_train))
plt.title("Sal vs exper")
plt.xlabel("Experience")
plt.ylabel("Salary in $")

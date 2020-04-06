import pandas as pd
import numpy as np
df = pd.read_csv('insurance.csv')



'''
correlation = df.corr()

#==============================
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(correlation )
plt.xticks(range(len(correlation.columns)),correlation.columns)
plt.yticks(range(len(correlation.columns)),correlation.columns)
sns.countplot(x=df['sex'])
#==============================

df.age.hist()



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
x_lable = LabelEncoder()
he = OneHotEncoder()
df['sex'] = x_lable.fit_transform(df['sex'])
df['smoker'] = x_lable.fit_transform(df['smoker'])
df['region'] = x_lable.fit_transform(df['region'])

'''
'''
import seaborn as sns
sns.pairplot(df, x_vars=['age','sex','bmi','smoker','children','region'], y_vars='charges', size=4, aspect=0.7)

df =pd.get_dummies(df,columns=['sex','smoker','region'])



x = df.drop(columns=['charges'])
y = df['charges']

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.3,random_state=42)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtr,ytr)
ypred = lr.predict(xte)
'''



# regression models
 # for feature scaling
from sklearn.pipeline import Pipeline # for using pipeline
 # for linear regression
 # for adding polynomial features
from sklearn.linear_model import Ridge# for ridge regression
from sklearn.linear_model import Lasso # for lasso regression
 # for support vector regression
 # for decisiton tree regression
 # for random forest regression
# hyptertuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
# extra


from collections import Counter
from IPython.core.display import display, HTML
#sns.set_style('darkgrid')



import matplotlib.pyplot as plt
import seaborn as sns
corr = df.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 8))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()


df = pd.get_dummies(df,columns=['sex','smoker','region'])
x = df.drop(columns=['charges','sex_male','smoker_no','region_northeast'])
y = df.iloc[:,[3]].values

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.3,random_state=0)


#====================================================Linear Regression==================
from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(xtr,ytr)
ypred = Lin_reg.predict(xte)
ypre_tr = Lin_reg.predict(xtr)

from sklearn.model_selection import cross_val_score
cv_lin = cross_val_score(estimator = Lin_reg , X=x, y= y , n_jobs=-1 , cv = 10 )
cv_lin.mean()
cv_lin.max()
cv_lin.min()


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

r2_reg = r2_score(yte,ypred)
r2_score(ytr,ypre_tr)
mse_lin = mean_squared_error(yte,ypred)
rmse_lin = np.sqrt(mse_lin)

#============================================Polynomial Regression ===================
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly_x = poly.fit_transform(xtr)
poly_xtest = poly.fit_transform(xte)
x_fulpoly = poly.fit_transform(x) #transformin polynomial feature

poly_reg = LinearRegression()
poly_reg.fit(poly_x,ytr)


ypre_poly = poly_reg.predict(poly_xtest)
ypre_train_poly = poly_reg.predict(poly_x)


from sklearn.model_selection import cross_val_score
cv_poly = cross_val_score(estimator = poly_reg , X = x_fulpoly , y = y, n_jobs=-1 , cv =10)
cv_poly.mean()
cv_poly.max()
cv_poly.min()

r2_poly = r2_score(yte,ypre_poly)
r2_score(ytr,ypre_train_poly)
mse_pol = mean_squared_error(yte,ypre_poly)
rmse = np.sqrt(mse_pol)

#==============================================SVM Regression ====================

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrrrr = sc.fit_transform(xtr)
ytrrrr = sc.fit_transform(ytr)
xteeee = sc.fit_transform(xte)
yteeee = sc.fit_transform(yte)
svr = SVR()

params ={'kernel':['rbf', 'sigmoid'],
         'degree':[1,2,3,4],
         'gamma':[0.1,1,'scale', 'auto'],
         'tol' : [1e-3,1e-2,0.01,0.1],
         'C':[0.001,0.01,0.1,1]}


estmator = RandomizedSearchCV(estimator=svr,param_distributions=params,n_iter = 20 ,n_jobs =-1,cv=10)
estmator.fit(x,y)
estmator.best_estimator_

svr = SVR(C=1, cache_size=200, coef0=0.0, degree=4, epsilon=0.1, gamma=0.1,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.01, verbose=False)

svr.fit(xtrrrr,ytrrrr)
yppp = svr.predict(xteeee)
re_svm = r2_score(yteeee,yppp)
mse_svm = mean_squared_error(sc.inverse_transform(yteeee),sc.inverse_transform(yppp)) #invers StandardScaler
rmse_svm = np.sqrt(mse_svm)


#==================================================decision Tree============================

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

x_main = sc.fit_transform(x)
y_main = sc.fit_transform(y)

Tree_params ={ "max_depth": np.arange(1,21),
              "min_samples_leaf": [1, 5, 10, 20, 50, 100],
              "min_samples_split": np.arange(2, 11),
              "criterion": ["mse"],
              "random_state" : [42]}
treeest= RandomizedSearchCV(estimator=dtr,param_distributions=Tree_params,n_iter=15,n_jobs=-1,cv=20)
treeest.fit(x_main,y_main)
treeest.best_score_

#dtr.fit(xtr,ytr)
#ypre_tre = dtr.predict(xte)
#r2_tree=r2_score(yte,ypre_tre)

dtr= DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=6,
                      max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=5, min_samples_split=3,
                      min_weight_fraction_leaf=0.0, presort='deprecated',
                      random_state=42, splitter='best')


dtr.fit(xtrrrr,ytrrrr)
ypre_tre = dtr.predict(xteeee)
r2_tree=r2_score(yteeee,ypre_tre)
rms = mean_squared_error(yte,sc.inverse_transform(ypre_tre))
rrmse = np.sqrt(rms)


#===============================random Forest Regression ======================

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_jobs=-1)

params_forest = {'n_estimators':list(range(50,500,25)), 
          'criterion':["mse"],
          'min_samples_split':[2,3,4] ,
          'min_samples_leaf':[1,2,3,4,5,6], 
          'max_features':list(range(1,12)), 
          'max_leaf_nodes':[2,3,4,5,6,7,8],
          'bootstrap':['True'],
          'random_state':[42]}

forest_est = RandomizedSearchCV(estimator=rfr, param_distributions=params_forest,n_iter=15,n_jobs=-1,cv=10)
forest_est.fit(x_main,y_main)
forest_est.best_score_

rfr2 = RandomForestRegressor(bootstrap='True', ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features=6, max_leaf_nodes=8,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=2,
                      min_samples_split=4, min_weight_fraction_leaf=0.0,
                      n_estimators=350, n_jobs=-1, oob_score=False,
                      random_state=42, verbose=0, warm_start=False)
rfr2.fit(xtrrrr,ytrrrr)
ypre_forest = rfr2.predict(xteeee)
r2_forest = r2_score(yteeee,ypre_forest)
rms_forest = mean_squared_error(yte,sc.inverse_transform(ypre_forest))
rmse_forest = np.sqrt(rms_forest)

print ("LinearRegression Score : {}".format(r2_reg))
print ("LinearRegression RootMeanSquareerror : {}".format(rmse_lin))
print ("Poly Score : {}".format(r2_poly))
print ("Poly RootMeanSquareerror : {}".format(rmse))
print ("SVM Score : {}".format(re_svm))
print ("SVM RootMeanSquareerror : {}".format(rmse_svm))
print ("DTree Score : {}".format(r2_tree))
print ("DTree RootMeanSquareerror : {}".format(rrmse))
print ("RandomForest Score : {}".format(r2_forest))
print ("RandomForest RootMeanSquareerror : {}".format(rmse_forest))
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Data
Data = pd.read_csv('Data.csv')
X = Data.iloc[:,:-1].values
Y = Data.iloc[:,3].values

#taking care of missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#categorising the data to numarical value
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X =    LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder()
X= onehotencoder.fit_transform(X).toarray()

labelencoder_Y =    LabelEncoder()
Y = labelencoder_X.fit_transform(Y)

#spltting the dataset in to train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 0.33, random_state = 0)


#feature scaling 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
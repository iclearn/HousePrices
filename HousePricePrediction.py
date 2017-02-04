# -*- coding: utf-8 -*-
# imports
import pandas  as pd
import seaborn as sns
import numpy as np
from math import sqrt 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

####################################################
# load data
train = pd.read_csv("~/Downloads/train.csv")
test = pd.read_csv("~/Downloads/test.csv")

####################################################
# exploration
# look at the structure of training data
train.shape
test.shape
train.head(5)
test.head(5)

# finding correlations in training data
train.corr()["SalePrice"]

# plotting
sns.regplot(train["OverallQual"], train["SalePrice"], lowess = True)
sns.regplot(train["GrLivArea"], train["SalePrice"], lowess = True)

####################################################
# preprocessing the data - both train and test

# all data
data = pd.DataFrame.append(train, test)

# dealing with NAs
# determine number of NA in columns
missing = data.isnull().sum()
missing[missing>0]

# Alley
data.loc[:, "Alley"] = data.loc[:, "Alley"].fillna("No")

# BsmtQual etc
data.loc[:, "BsmtQual"] = data.loc[:, "BsmtQual"].fillna("No")
data.loc[:, "BsmtCond"] = data.loc[:, "BsmtCond"].fillna("No")
data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna("No")
data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna("No")
data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna("No")

# Electrical
data.loc[:, "Electrical"] = data.loc[:, "Electrical"].fillna("No")

# Fence
data.loc[:, "Fence"] = data.loc[:, "Fence"].fillna("No")
        
# FireplaceQu
data.loc[:, "FireplaceQu"] = data.loc[:, "FireplaceQu"].fillna("No")
data.loc[:, "Fireplaces"] = data.loc[:, "Fireplaces"].fillna(0)

# GarageType etc
data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna("No")
data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna("No")
data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna("No")
data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna("No")
data.loc[:, "GarageYrBlt"] = data.loc[:, "GarageYrBlt"].fillna(0)

# LotFrontage
replace = data["LotFrontage"].median()
data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(replace)

# MasVnrType
data.loc[:, "MasVnrType"] = data.loc[:, "MasVnrType"].fillna("None")
data.loc[:, "MasVnrArea"] = data.loc[:, "MasVnrArea"].fillna(0)

# MiscFeature
data.loc[:, "MiscFeature"] = data.loc[:, "MiscFeature"].fillna("No")

# PoolQC
data.loc[:, "PoolQC"] = data.loc[:, "PoolQC"].fillna("No")

# Some numerical features are actually really categories
data = data.replace({"MSSubClass" : {20 : "MS20", 30 : "MS30", 40 : "MS40", 45 : "MS45", 
                                       50 : "MS50", 60 : "MS60", 70 : "MS70", 75 : "MS75", 
                                       80 : "MS80", 85 : "MS85", 90 : "MS90", 120 : "MS120", 
                                       150 : "MS150", 160 : "MS160", 180 : "MS180", 190 : "MS190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })

# Encode some categorical features as ordered numbers when there is information in the order
data = data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )

# one hot encode categorical values
categoricalFeatures = data.select_dtypes(include = ["object"]).columns
dataCat = data[categoricalFeatures]
dataCat = pd.get_dummies(dataCat)

# dealing with outliers
numerical_features = data.select_dtypes(exclude = ["object"]).columns
dataNum = data[numerical_features]

# Join categorical and numerical features
data = pd.concat([dataNum, dataCat], axis = 1)


###########################################################################
# again split to train and test data after transformation
train = data[0:1460]

# removing outliers
train = train[train.GrLivArea < 4000]

test = data[1460:2919]
del test["SalePrice"]

# Handle remaining missing values for numerical features by using median as replacement
test = test.fillna(test.median())

# remove id column
del train["Id"]

# transform saleprice
train["SalePrice"] = np.log1p(train["SalePrice"])

# splitting to training vectors and target vector
target = train["SalePrice"]
del train["SalePrice"]

# split training data to training and validation set
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.3, random_state = 0)

###########################################################################
# training model
regressionModel = GradientBoostingRegressor(n_estimators=500, max_depth=5, loss='lad')
regressionModel.fit(X_train, y_train)

# root mean square error
rmse = sqrt(mean_squared_error((y_test), (regressionModel.predict(X_test))))
print("MSE: %.4f" % rmse)

###########################################################################
# for kaggle submission
ids = test["Id"]
del test["Id"]
predictions = np.expm1(regressionModel.predict(test))
submission = pd.DataFrame({'Id' : ids,'SalePrice' : predictions})
submission.to_csv("gbm.csv", sep=',', encoding='utf-8', index = False)

###########################################################################
# model 2 - lasso regression 
lassoModel = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 150000, cv = 10)
lassoModel.fit(X_train, y_train)

# root mean square error
rmse = sqrt(mean_squared_error((y_test), (lassoModel.predict(X_test))))
print("MSE: %.4f" % rmse)

##########################################################################
# for kaggle submission
predictions = np.expm1(lassoModel.predict(test))
submission = pd.DataFrame({'Id' : ids,'SalePrice' : predictions})
submission.to_csv("lasso.csv", sep=',', encoding='utf-8', index = False)

###########################################################################

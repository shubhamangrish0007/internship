#======================= IMPORT PACKAGES ============================

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing


#======================= DATA SELECTION =========================

print("=======================================")
print("---------- Data Selection -------------")
print("=======================================")
data=pd.read_csv('train.csv')
print(data.head(10))
print()

#==================== PREPROCESSING =======================================

#checking missing values

print("=====================================================")
print("--------- Before Checking missing values ------------")
print("=====================================================")
print(data.isnull().sum())
print()


print("=====================================================")
print("--------- After Checking missing values ------------")
print("=====================================================")
data=data.fillna(0)
print(data.isnull().sum())
print()


#==== LABEL ENCODING ====

label_encoder = preprocessing.LabelEncoder() 
print("------------------------------------------------------")
print(" Before label encoding ")
print("------------------------------------------------------")
print()
print(data['SaleCondition'].head(10))

print("------------------------------------------------------")
print("After label encoding ")
print("------------------------------------------------------")
print()

data= data.astype(str).apply(label_encoder.fit_transform)

print(data['SaleCondition'].head(10))


#========================= DATA SPLITTING ============================

#=== TEST AND TRAIN ===

x=data.drop('SalePrice',axis=1)
y=data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print("-----------------------------------------------------------")
print("======================= Data splitting ====================")
print("-----------------------------------------------------------")
print()
print("Total No Of data          :",data.shape[0])
print()
print("Total No of Training data :",X_train.shape[0])
print()
print("Total No of Testing data :",X_test.shape[0])
print()


#========================= CLASSIFICATION ============================

from sklearn.linear_model import Ridge
from sklearn import metrics

#=== ridge regression ===

#initialize the model
ridgeR = Ridge(alpha = 1)

#fitting the model
ridgeR.fit(X_train, y_train)

#predict the model
y_pred = ridgeR.predict(X_test)


print("-----------------------------------------------------------")
print("======================= RIDGE REGRESSION ==================")
print("-----------------------------------------------------------")
print()


mae_ridge=metrics.mean_absolute_error(y_test, y_pred)

print("1.Mean Absolute Error       : ",mae_ridge)
print()
mse_ridge=metrics.mean_squared_error(y_test, y_pred)/1000
print("2.Mean Squared Error       : ",mae_ridge)
print()
import numpy as np
rmse_rid=np.sqrt(mse_ridge)
print("3.Root Mean Squared Error  : ",rmse_rid)



#=== lasso regression ===

from sklearn.linear_model import Lasso

#initialize the model
lasso = Lasso(alpha = 1)

#fitting the model
lasso.fit(X_train, y_train)

#predict the model
y_pred = lasso.predict(X_test)


print("-----------------------------------------------------------")
print("======================= LASSO REGRESSION ==================")
print("-----------------------------------------------------------")
print()


mae_lasso=metrics.mean_absolute_error(y_test, y_pred)

print("1.Mean Absolute Error       : ",mae_lasso)
print()
mse_lasso=metrics.mean_squared_error(y_test, y_pred)/1000
print("2.Mean Squared Error       : ",mse_lasso)
print()
import numpy as np
rmse_las=np.sqrt(mse_lasso)
print("3.Root Mean Squared Error  : ",rmse_las)


#========================= PREDICTION ============================

print("-----------------------------------------------------------")
print("======================= PREDICTION ========================")
print("-----------------------------------------------------------")
print()

for i in range(0,10):
    Results=y_pred[i]
    print("------------------------------------------")
    print()
    print([i],"The predicted house price is ", Results)
    print()

#============================ IMPORT PACKAGES ===============================

import pandas as pd
import re
from sklearn.model_selection import train_test_split

#============================ 1. DATA SELECTION ===========================

print("-------------------------------------------")
print(" DATA SELECTION")
print("-------------------------------------------")
print()

data_frame=pd.read_excel("customer_retention_dataset.xlsx")
print(data_frame.head(20))
print()



#==================== 2.PREPROCESSING =======================================

#==== checking missing values ====

print("-------------------------------------------")
print("BEFORE HANDLING MISSING VALUES")
print("-------------------------------------------")
print()
print(data_frame.isnull().sum())

#==== Label Encoding ====

from sklearn import preprocessing

print("------------------------------------------------")
print(" BEFORE LABEL ENCODING ")
print("------------------------------------------------")
print()
print(data_frame['3 Which city do you shop online from?'].head(20))


label_encoder = preprocessing.LabelEncoder()

data_frame['3 Which city do you shop online from?'] = label_encoder.fit_transform(data_frame['3 Which city do you shop online from?'])

print("-------------------------------------------")
print(" AFTER LABEL ENCODING ")
print("------------------------------------------")
print()
print(data_frame['3 Which city do you shop online from?'].head(20))


#3 Which city do you shop online from?          

#========================= 4.DATA SPLITTING ============================

x=data_frame.drop(['47 Getting value for money spent'],axis=1)
y=data_frame['47 Getting value for money spent']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print("-----------------------------------------------------------")
print("DATA SPLITTING")
print("-----------------------------------------------------------")
print()
print("Total No of input data    :",data_frame.shape[0])
print()
print("Total No of training data :",X_train.shape[0])
print()
print("Total No of testing data  :",X_test.shape[0])
print()


#47 Getting value for money spent

#========================= 5.CLASSIFICATION ============================

# === KNN ==

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

#fitting the model
knn.fit(X_train, y_train)

#predict the model
nb_pred = knn.predict(X_train)

from sklearn.metrics import accuracy_score

acc_nb=accuracy_score(y_train, nb_pred)*100

print("----------------------------")
print(" K - NEAREST NEIGHBOUR ")
print("----------------------------")
print()
print("1. ACCURACY = ", acc_nb,'%')
print()
print("2. Classification Report")
print()
from sklearn import metrics
print(metrics.classification_report(y_train, nb_pred))


# === NAIVE BAYES ===

from sklearn.tree import DecisionTreeClassifier 

#initialize the model
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=100, min_samples_leaf=1)

#fitting the model
dt.fit(X_train, y_train)

#predict the model
dt_prediction=dt.predict(X_test)

acc_dt=accuracy_score(y_test, dt_prediction)*100

print("----------------------------")
print(" DECISION TREE ")
print("----------------------------")
print()
print("1. ACCURACY = ", acc_dt,'%')
print()
print("2. Classification Report")
print()
print(metrics.classification_report(y_test, dt_prediction))


print()
print("---------------------------------------------------")
print()
import matplotlib.pyplot as plt
vals=[acc_nb,acc_dt]
inds=range(len(vals))
labels=["NB ","DT "]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title('Comparison graph--ACC')
plt.show() 
                                                                                                                                                                       
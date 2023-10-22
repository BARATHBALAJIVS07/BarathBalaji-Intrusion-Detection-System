# STAGE 1
print("STAGE 1")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,cross_val_score
#from keras.models import Sequential
from keras.layers import Dense 
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder

print("STAGE 2")




df=pd.read_csv('dataset.csv')
df_value=df[' Label'].value_counts()
df[' Label']=df[' Label'].apply({'DoS Hulk':'DoS', 'DoS GoldenEye':'DoS','DoS Slowhttptest':'DoS','DoS slowloris':'DoS' ,'BENIGN':'BENIGN' ,'DDoS':'DDoS', 'PortScan':'PortScan'}.get)
df2=df.drop_duplicates() 
df2_value=df2[' Label'].value_counts()
datatype=df2.dtypes 
df2['Flow Bytes/s']=df2['Flow Bytes/s'].astype('float64')
df2[' Flow Packets/s']=df2[' Flow Packets/s'].astype('float64')
NaN_values=df2.isnull().sum() 
df2['Flow Bytes/s'].fillna(df2['Flow Bytes/s'].mean(),inplace=True)#
print('Datasetin  : \n',df_value)
print('Datasetin inlk (row,Column) sayısı: {} '.format(df.shape))
print('Datasetin Label DoS   ı:\n',df2_value)
print('Datasetin son (row,Column) sayısı: {} '.format(df2.shape))


print("STAGE 3")





dataset=pd.read_csv('dataset.csv')
dataset
DoS_df1=dataset[dataset[' Label']=='BENIGN']
DoS_df=DoS_df1.append(dataset[dataset[' Label']=='DoS'])
DoS_df
DDoS_df1=dataset[dataset[' Label']=='BENIGN']
DDoS_df=DDoS_df1.append(dataset[dataset[' Label']=='DDoS'])
DDoS_df

PortScan_df1=dataset[dataset[' Label']=='BENIGN']
PortScan_df=PortScan_df1.append(dataset[dataset[' Label']=='PortScan'])
PortScan_df
NA_df=dataset
NA_df[' Label']=NA_df[' Label'].apply({'DoS':'Anormal','BENIGN':'Normal' ,'DDoS':'Anormal', 'PortScan':'Anormal'}.get)
NA_df


print("STAGE 4")


def train_test_dataset(df):
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop([' Label'],axis=1) 
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 0, stratify = y)
    return  X_train, X_test, y_train, y_test






def feature_selection(df):
    feature=(df.drop([' Label'],axis=1)).columns.values
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop([' Label'],axis=1) 
    Y = df.iloc[:, -1].values.reshape(-1,1)
    Y=np.ravel(Y)
    rf.fit(X, Y)
    print ("Features sorted by their score:")
    print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature), reverse=True))



#Feature
print("STAGE 5")
#feature_selection(dataset)


DoSX_train, DoSX_test, DoSy_train, DoSy_test=train_test_dataset(DoS_df) 
#DDoSX_train, DDoSX_test, DDoSy_train, DDoSy_test=train_test_dataset(DDoS_df)
#PS_X_train,PS_X_test,PS_y_train, PS_y_test=train_test_dataset(PortScan_df)
#NA_X_train, NA_X_test, NA_y_train, NA_y_test=train_test_dataset(NA_df)

print("STAGE 6")
X_train=DoSX_train
X_test=DoSX_test
y_train=DoSy_train
y_test=DoSy_test
n_signals = 1 
n_outputs = 1 
print("STAGE 7")
#Build the model
verbose, epochs, batch_size = True, 15, 16
n_steps, n_length = 40, 10



model = Sequential()

# Adding the input layer and the ACNN layer
model.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the ACNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


model.fit(X_train, y_train, batch_size = 32, epochs = 2)


print("STAGE 9")
regressor.save('model1.h5')
print("STAGE 10")


# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=0)


# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

print("STAGE 11")

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


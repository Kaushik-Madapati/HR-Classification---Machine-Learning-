
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[2]:


df = pd.read_csv('train_HRSet.csv')


# In[3]:


df.head()


# In[4]:


from keras.utils.np_utils import to_categorical


# In[5]:


def edu_label(education):
    if education == "Bachelor's" :
        return 1
    elif education == "Master's & above" :
        return 2
    else :
        return 3


# In[6]:


def dpt_label(education):
    if education == "Sales & Marketing" :
        return 1
    elif education == "Operations" :
        return 2
    elif education == "Technology" :
        return 3
    elif education == "Procurement" :
        return 4
    elif education == "Analytics" :
        return 5
    elif education == "HR" :
        return 6
    elif education == "Legal" :
        return 7
    else :
        return 8


# In[7]:


def DataCleaning(data) :
    data['education'] = data['education'].apply(edu_label )
   # data['department'] = data['department'].apply(dpt_label )
    
    department = pd.get_dummies(data['department'], drop_first=True)
    #region = pd.get_dummies(df['region'], drop_first=True)
    #education = pd.get_dummies(data['education'], drop_first =True)
    recruitment_channel  = pd.get_dummies(data['recruitment_channel'], drop_first =True)
    gender  = pd.get_dummies(data['gender'], drop_first =True)
    data = pd.concat([data, department, recruitment_channel,gender], axis =1)
    drop_items = ['recruitment_channel', 'department', 'gender']
    data.drop(drop_items, axis=1, inplace= True)
    data['region'] = data['region'].apply(lambda x: int(x.split('_')[1]))
    data['previous_year_rating']= data['previous_year_rating'].apply(lambda x: 3 if pd.isnull(x) else x)
    return data


# In[8]:


df['department'].value_counts()


# In[9]:


df.info()


# In[10]:


df['education'].value_counts()


# In[11]:


df.info()


# In[12]:


df.dropna()
X = df.drop(['employee_id', 'is_promoted'],axis=1)
X_clean = DataCleaning(X)
X_clean = scale(X_clean)
sc = StandardScaler();
#sc = MinMaxScaler();
X_Scaled =  sc.fit_transform(X_clean)

y = df['is_promoted'].values
y_cat = to_categorical(y)


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression 


# In[14]:


#X_train, X_test, y_train, y_test = train_test_split(X_clean,y_cat,test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(X_Scaled,y,test_size=0.3)
logicReg = LogisticRegression()
logicReg.fit(X_train,y_train)

y_pred = logicReg.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[15]:


import keras.backend as K
from keras.models import Sequential
from  keras.layers import Dense
from  keras.layers import Dropout
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_Scaled,y_cat,test_size=0.3,random_state = 42)

K.clear_session()
model = Sequential()
model.add(Dense(10, input_shape=(20,), activation='relu'))
model.add(Dense(2,  activation='softmax'))
model.compile(loss='categorical_crossentropy',  optimizer=Adam(lr=0.01), metrics=['accuracy'])

#model.add(Dense(40, input_shape =(20,), activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(2,  activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.5), metrics=['accuracy'])

#initial_weights = model.get_weights()
h = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.3)


# In[17]:


y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)


# In[18]:


from sklearn.metrics import classification_report, confusion_matrix


# In[19]:


print(classification_report(y_test_class, y_pred_class))
confusion_matrix(y_test_class, y_pred_class)


# In[20]:


test_data_raw = pd.read_csv('test_HRSet.csv')
test_data = test_data_raw.drop(['employee_id'],axis=1)
test_data_clean = DataCleaning(test_data)
test_data_scaled =  sc.fit_transform(test_data_clean)
#test_data_clean = scale(test_data_clean)
y_test_pred = model.predict(test_data_scaled)


# In[21]:


submission = pd.DataFrame({ "employee_id": test_data_raw["employee_id"],      
                            "is_promoted": np.argmax(y_test_pred, axis=1) })
    

submission.to_csv('HR_output_test2.csv', index=False)


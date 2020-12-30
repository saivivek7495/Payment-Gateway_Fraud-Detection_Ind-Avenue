#!/usr/bin/env python
# coding: utf-8

# In[126]:


# Detecting Fraud Transactions for "IndAvenue" Payment Gateway Startup


# In[1]:


# Importing Libraries 

import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[3]:


get_ipython().system('pip install imblearn')


# In[4]:


os.getcwd()


# In[5]:


os.chdir('C:\\Users\\SAI VIVEK.K\\MITH_Exam_Sep')


# In[6]:


# Reading and Importing the Dataset
Indav_train = pd.read_csv('train_data-1599717478076.csv')
Indav_test  = pd.read_csv('test_data-1599717650029.csv')


# In[7]:


# Exploratory Data Analysis and Preprocessing

# 1) Identification of Varibales and Datatypes


# In[8]:


Indav_train.dtypes


# In[9]:


print(Indav_train.shape)
print(Indav_test.shape)


# In[10]:


Indav_train.head()


# In[11]:


# Summary Statistics and Distribution of the Numeric Columns

Indav_train.describe()


# In[12]:


Indav_train.describe(include = 'object')


# In[13]:


# 2) Non - Graphical Univariate Analysis
#Distribution of dependent variable


# In[14]:


Indav_train.is_fraud.value_counts(normalize = True)*100


# In[15]:


# number of fraud and non-fraud observations in the dataset
frauds = len(Indav_train[Indav_train.is_fraud == 1])
nonfrauds = len(Indav_train[Indav_train.is_fraud == 0])
print("Frauds", frauds); print("Non-frauds", nonfrauds)


# In[17]:


Indav_train[Indav_train.is_fraud == 1].device_type.value_counts(normalize=True)*100


# In[18]:


Indav_train[Indav_train.is_fraud == 1].payment_method.value_counts(normalize=True)*100


# In[19]:


Indav_train[Indav_train.is_fraud == 1].partner_category.value_counts(normalize=True)*100


# In[20]:


pd.set_option('display.max_rows', None)


# In[16]:


# Findind Null Values in the Dataframe

Indav_train.isnull().sum()


# In[21]:


Indav_train.groupby(['payment_method','is_fraud']).mean()


# In[22]:


Indav_train.groupby(['money_transacted','is_fraud']).mean()


# In[23]:


#3) Graphical Univariate Analysis
#Plot the Distribution


# In[24]:


plt.figure(figsize=(4,6))
sns.countplot(x="is_fraud", data= Indav_train)
plt.show()

Indav_train.is_fraud.value_counts()


# In[25]:


plt.figure(figsize=(14,12))
sns.countplot(x="payment_method", data= Indav_train)
plt.show()

Indav_train.payment_method.value_counts()


# In[26]:


plt.figure(figsize=(14,12))
sns.countplot(x="partner_category", data= Indav_train)
plt.show()

Indav_train.partner_category.value_counts()


# In[28]:


categorical_columns = [cname for cname in Indav_train.columns if Indav_train[cname].dtype == 'object']

numerical_columns   = [cname for cname in Indav_train.columns if Indav_train[cname].dtype in['int64', 'float64']]


# In[29]:


plt.figure(figsize=(12,10))
data = Indav_train[numerical_columns]

df = pd.DataFrame(data)

corrMatrix = df.corr(method = 'pearson',min_periods = 1)
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[30]:


# Feature Engineering


# In[ ]:


#Dropping Unnecessary Columns
Indav_train.drop("partner_id", axis = 1, inplace= True)
Indav_test.drop("partner_id", axis = 1, inplace= True)


# In[32]:


#Type Casting

col = ['payment_method', 'partner_category','country','device_type','is_fraud']
Indav_train[col] = Indav_train[col].astype('category')


# In[33]:


Indav_train['money_transacted'] = Indav_train['money_transacted'].astype('int64')
Indav_test['money_transacted'] = Indav_test['money_transacted'].astype('int64')


# In[34]:


Indav_train['transaction_initiation'] = pd.to_datetime(Indav_train['transaction_initiation']).dt.year
Indav_test['transaction_initiation'] = pd.to_datetime(Indav_test['transaction_initiation']).dt.year


# In[35]:


Indav_train.dtypes


# In[36]:


Indav_train.head()


# In[37]:


print(Indav_train.shape)
print(Indav_test.shape)


# In[38]:


#Split Numeric and Categorical Columns


# In[39]:


cat_attr = list(Indav_train.select_dtypes("category").columns)
num_attr = list(Indav_train.columns.difference(cat_attr))


# In[40]:


cat_attr.pop()


# In[41]:


cat_attr


# In[42]:


num_attr


# In[62]:


cols_to_use = cat_attr + num_attr


# In[43]:


#Instantiate Pre-processing Objects for Pipeline


# In[44]:


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_attr),
        ('cat', categorical_transformer, cat_attr)])


# In[45]:


# Undersampling for handling Class imbalance


# In[46]:


#Number of samples which are Fraud
no_frauds = len(Indav_train[Indav_train['is_fraud'] == 1])


# In[47]:


# indices of non fraud samples
non_fraud_indices = Indav_train[Indav_train.is_fraud == 0].index


# In[48]:


#Random sample non fraud indices
random_indices = np.random.choice(non_fraud_indices,no_frauds, replace=False)


# In[49]:


fraud_indices = Indav_train[Indav_train.is_fraud == 1].index


# In[51]:


#Concat fraud indices with sample non-fraud ones
under_sample_indices = np.concatenate([fraud_indices,random_indices])


# In[52]:


#Get Balance Dataframe
under_sample = Indav_train.loc[under_sample_indices]


# In[152]:


#Train_Test_Split
X_under = under_sample.loc[:,under_sample.columns != 'is_fraud']
y_under = under_sample.loc[:,under_sample.columns == 'is_fraud']
X_under_train, X_under_valid, y_under_train, y_under_valid = train_test_split(X_under,y_under,test_size = 0.1, random_state = 0)


# In[153]:


print('X_train dimensions: ', X_under_train.shape)
print('y_train dimensions: ', y_under_train.shape)
print('X_test dimensions:  ', X_under_valid.shape)
print('y_test dimensions:  ', y_under_valid.shape)


# In[154]:


y_under_train.is_fraud.value_counts()


# In[155]:


# Model Building


# In[156]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


# In[157]:


# Build a Logistic Regression Model


# In[158]:


clf_logreg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])


# In[159]:


get_ipython().run_cell_magic('time', '', 'clf_logreg.fit(X_under_train, y_under_train)')


# In[161]:


train_pred = clf_logreg.predict(X_under_train)
valid_pred = clf_logreg.predict(X_under_valid)

print(clf_logreg.score(X_under_train, y_under_train))
print(clf_logreg.score(X_under_valid, y_under_valid))
print("\n")
print(confusion_matrix(y_true= y_under_train, y_pred = train_pred))

confusion_matrix_test = confusion_matrix(y_true=y_under_test, y_pred =  test_pred)
confusion_matrix_test


# In[174]:


f1_score(y_true = y_under_valid, y_pred = valid_pred)


# In[163]:


# Build a Random Forest Model


# In[164]:


clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])


# In[165]:


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=143)

param_grid = {"classifier__n_estimators" : [150, 250, 300],
              "classifier__max_depth" : [5,8,10],
              "classifier__max_features" : [3, 5, 7],
              "classifier__min_samples_leaf" : [4, 6, 8, 10]}

rf_grid = GridSearchCV(clf, param_grid= param_grid, cv=kfold,verbose=1,n_jobs=6)


# In[166]:


get_ipython().run_cell_magic('time', '', 'rf_grid.fit(X_under_train,y_under_train)')


# In[167]:


rf_grid.best_params_


# In[168]:


Final_rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators = 10,
                                     max_features = 'sqrt',
                                     min_samples_leaf = 5,
                                     bootstrap = True,
                                     max_depth = 25,
                                     min_samples_split = 10))])


# In[169]:


get_ipython().run_cell_magic('time', '', 'Final_rf.fit(X_under_train, y_under_train)')


# In[172]:


train_pred = Final_rf.predict(X_under_train)
valid_pred = Final_rf.predict(X_under_valid)

print(Final_rf.score(X_under_train, y_under_train))
print(Final_rf.score(X_under_valid, y_under_valid))
print("\n")
print(confusion_matrix(y_true= y_under_train, y_pred = train_pred))

confusion_matrix_test = confusion_matrix(y_true=y_under_test, y_pred =  test_pred)
confusion_matrix_test


# In[173]:


f1_score(y_true = y_under_valid, y_pred = valid_pred)


# In[127]:


Final_X_test = Indav_test[cols_to_use]
#
test_pred = Final_rf.predict(Final_X_test)


# In[128]:


test_pred[0:10]


# In[129]:


submissions_df = pd.DataFrame({'transaction_number': Indav_test.transaction_number, 'is_fraud': test_pred})
submissions_df.to_csv("final_predictions.csv", index=None)
submissions_df.head()


# In[ ]:





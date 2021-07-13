#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pip install xlrd


# In[3]:


excelfile=pd.ExcelFile("TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx")

df_resign=pd.read_excel(excelfile,'Employees who have left')  # Sheet 2
df_exist=pd.read_excel(excelfile,'Existing employees')    # Sheet 3


# In[4]:


#Data Understanding
df_resign.head()


# In[65]:


df_resign.describe()


# In[5]:


df_exist.head()


# In[66]:


df_exist.describe()


# In[6]:


df_resign['dept'].value_counts()


# In[7]:


#Data Visualization
plt.figure(figsize=(15,10))
sns_plot99 = sns.catplot('dept',data=df_resign,kind='count',aspect=2)
# fig = sns_plot99.get_figure()


# In[67]:


df = pd.DataFrame({'Employee Status':['Existing', 'Left'], 'Number':[11429, 3572]})
ax = df.plot.bar(x='Employee Status', y='Number', rot=0)


# In[8]:


plt.figure(figsize=(16,10))
plt.title("Satisfication level vs last evaluation")
sns_plot = sns.scatterplot(x=df_resign['satisfaction_level'],y=df_resign['last_evaluation'],hue='number_project',data=df_resign)
fig = sns_plot.get_figure()
fig.savefig("figure1.png")


# In[68]:


heatmap1_data = pd.pivot_table(df_resign, values='satisfaction_level', 
                     index=['time_spend_company'], 
                     columns='salary')
sns.heatmap(heatmap1_data, cmap="YlGnBu", annot=True)


# In[9]:



plt.figure(figsize=(15,8))
plt.title('Salary vs Satisfaction level')
sns_plot1 = sns.boxplot(x=df_resign['salary'],y=df_resign['satisfaction_level'],hue='time_spend_company',data=df_resign,palette='Blues')
fig = sns_plot1.get_figure()
fig.savefig("figure2.png")


# In[10]:


plt.figure(figsize=(15,8))
plt.title("Salary vs Monthly hours spent")
sns.boxplot(x=df_resign['salary'],y=df_resign['average_montly_hours'],hue='number_project',palette='Blues',data=df_resign)
plt.show()


# In[11]:


plt.figure(figsize=(15,8))
plt.title("Average monthly hours vs promotions in last 5 years")
sns.boxplot(x=df_resign['promotion_last_5years'],y=df_resign['average_montly_hours'],hue='time_spend_company',data=df_resign,palette='Set3')
plt.show()


# In[69]:


plt.figure(figsize=(15,8))
plt.title('Average monthly hours vs number of projects')
sns.boxplot(x=df_resign['number_project'],y=df_resign['average_montly_hours'],data=df_resign,palette='Set3')
plt.show()


# In[70]:



plt.figure(figsize=(18,10))
plt.title("Department vs Satisfcation level")
sns.boxplot(x=df_resign['dept'],y=df_resign['satisfaction_level'],hue='time_spend_company',data=df_resign,palette='Set3')
plt.show()


# In[71]:


#Combining both the datasets into a single dataset
df_resign['Left'] = 1

df_exist['Left'] = 0

combined_df=pd.concat([df_resign,df_exist],axis=0)
# print(combined_df)
combined_df.head()


# In[15]:


combined_df.info()


# In[16]:



# Creating dummies

# columns=['dept','salary']

# dummies=pd.get_dummies(combined_df[columns],drop_first=True)

# combined_df=pd.concat([combined_df,dummies],axis=1)


# In[17]:


# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer

# cat_col = ['dept', 'salary']

# categorical_transformer = Pipeline(steps=[
#     ('onehotencoder',OneHotEncoder(handle_unknown='ignore'))
# ])

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# preprocessor = ColumnTransformer(transformers /


# In[18]:


# X_train.head()


# In[19]:



# combined_df=combined_df.drop(columns,axis=1)  # Dropping uncessary columns


# In[73]:


combined_df.head()


# In[74]:


combined_df.tail()


# In[72]:


combined_df.info()


# In[75]:


print("{0:.1f}% of people that have resigned from company X".format(100-(len(combined_df[combined_df['Left'] == 0])/len(combined_df))*100))


# In[22]:


# # Dividing the dataset into X and Y 
# combined_df.drop('Emp ID',inplace = True, axis = 1)
# X = combined_df.iloc[:, :-1]
# y = combined_df.iloc[:, -1]
# X.head()


# In[23]:


# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-2])], remainder='passthrough')
# ct2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
# X = np.array(ct1.fit_transform(X))
# X = np.array(ct2.fit_transform(X))
# # X = ct1.fit_transform(X)
# # X = ct2.fit_transform(X)


# In[24]:


# X = pd.DataFrame(data = X, index = combined_df.index, columns = combined_df.columns)


# In[25]:


# adjusting categorical columns

columns=['dept','salary']

dummies=pd.get_dummies(combined_df[columns],drop_first=True)
combined_df=pd.concat([combined_df,dummies],axis=1)

combined_df=combined_df.drop(columns,axis=1)


# In[26]:



# Dividing the dataset into X and Y 

X=combined_df.drop('Left',axis=1)

y=combined_df['Left']


# In[27]:


X.head()


# In[77]:


# Splitting of the X and y datasets into train and test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[78]:


X=pd.concat([X_train,y_train],axis=1)

emp_resign = X[X.Left==0]
emp_exist = X[X.Left==1]


# In[79]:


# X_train.drop('Emp ID', inplace = True, axis = 1)
# X_train.head()


# In[80]:


# Logisstic regression

from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import RFE

classifier1 = LogisticRegression()

# pipeline1 = Pipeline(steps = [
#     ('preprocessor',preprocessor),
#     ('classifier',classifier1)
# ])

# model=LogisticRegression()
# logreg=RFE(model,15)
# pipeline1.fit(X_train,y_train)
classifier1.fit(X_train.drop('Emp ID', axis = 1),y_train)

from sklearn.metrics import accuracy_score
predictions = classifier1.predict(X_test.drop('Emp ID', axis = 1))
predictions
# print("The Accuracy score using logistic regression is:{:.3f}".format(accuracy_score(y_test,classifier1.predict(X_test))))


# In[81]:



# Model evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("The Accuracy score using logistic regression is:{:.3f}".format(accuracy_score(y_test,classifier1.predict(X_test.drop('Emp ID', axis = 1)))))
print("The Precison score using logistic regression is:{:.3f}".format(precision_score(y_test,classifier1.predict(X_test.drop('Emp ID', axis = 1)))))
print("The Recall score using logistic regression is:{:.3f}".format(recall_score(y_test,classifier1.predict(X_test.drop('Emp ID', axis = 1)))))
print("The F1 score using logistic regression is:{:.3f}".format(f1_score(y_test,classifier1.predict(X_test.drop('Emp ID', axis = 1)))))


# In[82]:


# Random forest classifier

from sklearn.ensemble import RandomForestClassifier

classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

# pipeline2 = Pipeline(steps = [
#     ('preprocessor',preprocessor),
#     ('classifier',classifier2)
# ])

# # model=LogisticRegression()
# # logreg=RFE(model,15)
# pipeline2.fit(X_train,y_train)
# print("The Accuracy score using logistic regression is:{:.3f}".format(accuracy_score(y_test,pipeline2.predict(X_test))))
classifier2.fit(X_train.drop('Emp ID', axis = 1),y_train)

predictions2 = classifier2.predict(X_test.drop('Emp ID', axis = 1))
predictions2
# print("The Accuracy score using random forest classifer is:{:.3f}".format(accuracy_score(y_test,classifier2.predict(X_test))))


# In[83]:



# Model evaluation

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("The Accuracy score using Random Forest Classifier is:{:.3f}".format(accuracy_score(y_test,classifier2.predict(X_test.drop('Emp ID', axis = 1)))))
print("The Precison score using Random Forest Classifier is:{:.3f}".format(precision_score(y_test,classifier2.predict(X_test.drop('Emp ID', axis = 1)))))
print("The Recall score using Random Forest Classifier is:{:.3f}".format(recall_score(y_test,classifier2.predict(X_test.drop('Emp ID', axis = 1)))))
print("The Recall score using Random Forest Classifier is:{:.3f}".format(f1_score(y_test,classifier2.predict(X_test.drop('Emp ID', axis = 1)))))


# In[84]:


# Support vector classifier

from sklearn.svm import SVC

classifier3 = SVC(kernel = 'rbf', C = 1)

# pipeline3 = Pipeline(steps = [
#     ('preprocessor', preprocessor),
#     ('classifier', classifier3)
# ])

# model=LogisticRegression()
# logreg=RFE(model,15)
classifier3.fit(X_train,y_train)

predictions3 = classifier3.predict(X_test)
predictions3


# In[85]:


# Model evaluation

# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

print("The Accuracy score using SVC is:{:.3f}".format(accuracy_score(y_test,classifier3.predict(X_test))))
print("The Precison score using SVC is:{:.3f}".format(precision_score(y_test,classifier3.predict(X_test))))
print("The Recall score using SVC is:{:.3f}".format(recall_score(y_test,classifier3.predict(X_test))))


# In[86]:


# building with random forest classification as it is best suited

pred_h = np.concatenate((predictions2.reshape(len(predictions2),1),y_test.values.reshape(len(y_test),1)),1)
print(pred_h)


# In[87]:


# employees_prone_to_leave = []
# for emp_id, i in enumerate(pred_h):
#     if (i[0]!=i[-1] and i[0]==1):
#         employees_prone_to_leave.append(emp_id+1)


# employees_prone_to_leave#.extend([x for x in y_train if ])
# # print(len(pred_h))


# In[88]:


# Oversamlpling

# from sklearn.utils import resample

# y = combined_df['Left']
# X= combined_df.drop(['Left'],axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.75,random_state=50)

# X=pd.concat([X_train,y_train],axis=1)

# emp_not_left=X[X.Left==0]
# emp_left=X[X.Left==1]


# In[89]:


# unsampling the minority by adding dummy rows to the left equal to 1 

# left_upsampled= resample(emp_left,replace=True,n_samples=len(emp_not_left),random_state=50)

# left_upsampled=pd.concat([emp_not_left,left_upsampled])


# In[90]:


# left_upsampled.Left.value_counts()  # Both classes now having equal samples


# In[91]:



# # Preparing for X train and Y train dataset

# y_train=left_upsampled.Left
# X_train=left_upsampled.drop('Left',axis=1)


# In[92]:


# Model building

# new_logreg=LogisticRegression()
# logreg_rfe=RFE(new_logreg,15)
# logreg_rfe.fit(X_train.drop('Emp ID',axis=1),y_train)
# upsampled_pred=logreg_rfe.predict(X_test.drop('Emp ID',axis=1))


# In[93]:


# # Model evaluation

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# print("The Accuracy score using logistic regression is:{:.3f}".format(accuracy_score(y_test,upsampled_pred)))
# print("The Precison score using logistic regression is:{:.3f}".format(precision_score(y_test,upsampled_pred)))
# print("The Recall score using logistic regression is:{:.3f}".format(recall_score(y_test,upsampled_pred)))
# print("The F1 score using logistic regression is:{:.3f}".format(f1_score(y_test,upsampled_pred)))


# In[94]:



# # Model building

# rfc_upsampled=RandomForestClassifier()
# rfc_upsampled.fit(X_train.drop('Emp ID',axis=1),y_train)
# upsampled_rfc_pred=rfc_upsampled.predict(X_test.drop('Emp ID',axis=1))


# In[95]:


# Model evaluation

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# print("The Accuracy score using Random Forest Classifier is:{:.3f}".format(accuracy_score(y_test,upsampled_rfc_pred)))
# print("The Precison score using Random Forest Classifier is:{:.3f}".format(precision_score(y_test,upsampled_rfc_pred)))
# print("The Recall score using Random Forest Classifier is:{:.3f}".format(recall_score(y_test,upsampled_rfc_pred)))
# print("The F1 score using Random Forest Classifier is:{:.3f}".format(f1_score(y_test,upsampled_rfc_pred)))


# In[96]:



# # Model Building

# upsampled_svc=SVC(C=1)
# upsampled_svc.fit(X_train.drop('Emp ID',axis=1),y_train)
# svc_upsampled_pred=upsampled_svc.predict(X_test.drop('Emp ID',axis=1))


# In[97]:


# # Model evaluation

# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# print("The Accuracy score using SVC is:{:.3f}".format(accuracy_score(y_test,svc_upsampled_pred)))
# print("The Precison score using SVC is:{:.3f}".format(precision_score(y_test,svc_upsampled_pred)))
# print("The Recall score using SVC is:{:.3f}".format(recall_score(y_test,svc_upsampled_pred)))
# print("The F1 score using SVC is:{:.3f}".format(f1_score(y_test,svc_upsampled_pred)))


# In[98]:


# Random Forest

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

rfc=RandomForestClassifier(random_state=50)

n_folds=KFold(n_splits=5,shuffle=True, random_state=50)

parameters={'criterion':['gini','entropy'],'max_depth': range(5,30,5),'max_features': range(10,18,2),
            'min_samples_split': range(2,10,2)}

model_cv = GridSearchCV(estimator=classifier2,param_grid=parameters,cv=n_folds,verbose=1,
                        return_train_score=True,scoring='recall')


# In[99]:


model_cv.fit(X_train,y_train)


# In[100]:


model_cv.best_params_


# In[101]:


model_cv.best_score_


# In[102]:


final_classifier=RandomForestClassifier(criterion='entropy', max_depth=5, max_features=14, min_samples_split=2, random_state=0)
final_classifier.fit(X_train.drop('Emp ID',axis=1),y_train)
y_pred=final_classifier.predict(X_test.drop('Emp ID',axis=1))


# In[103]:


# #model evaluation
# from sklearn.metrics import classification_report

# print(classification_report(y_test,y_pred))


# In[104]:


# final_classifier.feature_importances_


# In[105]:


# X_train.columns


# In[106]:



# features=np.array(X_train.drop('Emp ID',axis=1).columns)
# important=final_rfc.feature_importances_
# indexes_features=important.argsort()
# for i in indexes_features:
#     print("{} : {:.2f}%".format(features[i],important[i]*100))


# In[107]:


# Finding employees who are prone to leave

y_test1=pd.concat([y_test,X_test['Emp ID']],axis=1)
y_test3=pd.DataFrame(y_pred)

y_test3.reset_index(inplace=True, drop=True)

gf=pd.concat([y_test1.reset_index(),y_test3],1)

new_df=gf[gf.Left==0]

new_df=new_df.drop('index',axis=1)

new_df.columns=['Left','Emp ID','Predicted_left']

Employees_prone_to_leave=new_df[new_df['Predicted_left']==1]
Employees_prone_to_leave=Employees_prone_to_leave.reset_index()
Employees_prone_to_leave=Employees_prone_to_leave.drop(['Left','Predicted_left','index'],axis=1)


# In[108]:


Employees_prone_to_leave


# In[109]:


result = []
for i in Employees_prone_to_leave.values:
    for j in i:
        result.append(j)
result


# In[110]:


output = pd.DataFrame({'Emp ID': result})
output.to_csv('submission.csv', index=False)


# In[111]:


output


# In[112]:


#Accuracy Check
print("The Accuracy score using final classifier is:{:.3f}".format(accuracy_score(y_test,y_pred)))
print("The Precison score using final classifier is:{:.3f}".format(precision_score(y_test,y_pred)))
print("The Recall score using final classifier is:{:.3f}".format(recall_score(y_test,y_pred)))


# In[ ]:





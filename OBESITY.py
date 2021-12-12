#!/usr/bin/env python
# coding: utf-8

# Importing the libraries and the data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro


# In[2]:


dataset = pd.read_csv('obesity_dataset.csv')


# Height and weight are highly correlated and they directly correlate to the BMI calc used for the target
# (Remove Height and Weight)

# In[3]:


dataset = dataset.drop(columns=['Height', 'Weight'])
print(dataset.shape)


# Missing Value Check

# In[4]:


sum(dataset.isnull().any(axis=1))


# Convert object/text variables to category variables

# In[5]:


columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]

for col in columns:
    dataset[col] = dataset[col].astype('category')


# Convert float to integer

# In[6]:


cols = ['FCVC', 'NCP', 'CH2O','TUE','FAF']
for col in cols:
    dataset[col] = round(dataset[col]).astype('int')


# Target Feature Distribution using Bar Chart

# In[7]:


vals = dataset['NObeyesdad'].value_counts()
vals_plt = vals.plot(kind='bar')
f = vals_plt.get_figure()
f.savefig('NObeyesdadBar.jpg')


# In[8]:


dataset.info()


# Encoding Yes/No values into Binary

# In[9]:


dataset.family_history_with_overweight.replace(('yes', 'no'), (1, 0), inplace=True)
dataset.FAVC.replace(('yes', 'no'), (1, 0), inplace=True)
dataset.SMOKE.replace(('yes', 'no'), (1, 0), inplace=True)
dataset.SCC.replace(('yes', 'no'), (1, 0), inplace=True)


# In[10]:


dataset.head()


# Age Distribution Plots

# In[11]:


a = sns.displot(dataset['Age'], kde =True)

a.savefig('agePlot.jpg') 


# In[12]:


dataset['Age'].mean()


# In[13]:


b = dataset.loc[:,['Age','NObeyesdad']]
b.groupby(['NObeyesdad']).mean()


# In[14]:


dataset = dataset.drop(columns=['Age'])


# In[15]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[16]:


rownames = dataset.columns
np.array(rownames)
ind = [1,2,3,4,6,7,8,9,10]
rows = []
for i in ind: 
    rows.append(rownames[i])
    
rows


# In[17]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,5,11,12])], remainder='passthrough')

X = pd.DataFrame(ct.fit_transform(X))
names = []
for i in ct.get_feature_names():
    if(i.__contains__("encoder")):
        names.append(i)
for i in rows: 
    names.append(i)
X.columns = names


# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# Chi-Squared Score

# In[19]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(13,'Score'))


# Importance Calculation using ExtraTressClassifier

# In[20]:


X = pd.DataFrame(X)
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
pl = feat_importances.nlargest(13).plot(kind='barh')
plt.show()
fig3 = pl.get_figure()
fig3.savefig('importance_features.jpg') 


# Splitting X and y into training and test sets %90-%10

# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)


# Data Scaling

# In[22]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Random Forest Classifier

# In[23]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Evaluation

# In[24]:


import sklearn.metrics as metrics
y_pred = classifier.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred) #confusion matrix
#Evaluation Metrics
accr = metrics.accuracy_score(y_test, y_pred)
prfs = metrics.precision_recall_fscore_support(y_test, y_pred)
print("Accuracy: {}".format(accr))
for i in prfs:
    print(i)    


# Confusion Matrix

# In[25]:


import itertools
with plt.style.context(('ggplot', 'seaborn')):
    fig = plt.figure(figsize=(6,6), num=1)
    plt.imshow(cm, interpolation='nearest',cmap= plt.cm.Blues )
    plt.xticks([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])
    plt.yticks([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i,cm[i, j], horizontalalignment="center",color="red")
    plt.grid(None)
    plt.title('Confusion Matrix')
    plt.colorbar();
    fig.savefig('conf_matrix.jpg')


# Cross Validation

# In[26]:


from sklearn.model_selection import cross_val_score, cross_validate
accuracies = cross_validate(estimator = classifier, X = X_train, y = y_train, cv = 10,scoring=('f1_macro','accuracy'))
print(accuracies['test_accuracy'].mean())
print(accuracies['test_f1_macro'].mean())
rand_acc = accuracies['test_accuracy']


# In[27]:


import scikitplot as skplt
pl = skplt.estimators.plot_learning_curve(classifier, X_train, y_train,
                                     cv=10, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Learning Curve");
plot_cross = pl.get_figure()
plot_cross.savefig('random_forest_cross.jpg')


# ## Decision Tree Classifier

# In[28]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', splitter="random" ,random_state = 0)
classifier.fit(X_train, y_train)


# Evaluation

# In[29]:


import sklearn.metrics as metrics
y_pred = classifier.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred) #confusion matrix
#Evaluation Metrics
accr = metrics.accuracy_score(y_test, y_pred)
prfs = metrics.precision_recall_fscore_support(y_test, y_pred)
print("Accuracy: {}".format(accr))
for i in prfs:
    print(i)    


# Confusion Matrix

# In[30]:


import itertools
with plt.style.context(('ggplot', 'seaborn')):
    fig = plt.figure(figsize=(6,6), num=1)
    plt.imshow(cm, interpolation='nearest',cmap= plt.cm.Blues )
    plt.xticks([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])
    plt.yticks([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i,cm[i, j], horizontalalignment="center",color="red")
    plt.grid(None)
    plt.title('Confusion Matrix')
    plt.colorbar();
    fig.savefig('conf_matrix.jpg')


# Cross Validation

# In[31]:


from sklearn.model_selection import cross_val_score, cross_validate
accuracies = cross_validate(estimator = classifier, X = X_train, y = y_train, cv = 10,scoring=('f1_macro','accuracy'))
print(accuracies['test_accuracy'].mean())
print(accuracies['test_f1_macro'].mean())
desc_acc = accuracies['test_accuracy']


# In[32]:


import scikitplot as skplt
pl = skplt.estimators.plot_learning_curve(classifier, X_train, y_train,
                                     cv=10, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Learning Curve");
plot_cross = pl.get_figure()
plot_cross.savefig('decision_cross.jpg')


# ## Support Vector Machine

# In[33]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# Evaluation

# In[34]:


import sklearn.metrics as metrics
y_pred = classifier.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred) #confusion matrix
#Evaluation Metrics
accr = metrics.accuracy_score(y_test, y_pred)
prfs = metrics.precision_recall_fscore_support(y_test, y_pred)
print("Accuracy: {}".format(accr))
for i in prfs:
    print(i)    


# Confusion Matrix

# In[35]:


import itertools
with plt.style.context(('ggplot', 'seaborn')):
    fig = plt.figure(figsize=(6,6), num=1)
    plt.imshow(cm, interpolation='nearest',cmap= plt.cm.Blues )
    plt.xticks([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])
    plt.yticks([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i,cm[i, j], horizontalalignment="center",color="red")
    plt.grid(None)
    plt.title('Confusion Matrix')
    plt.colorbar();
    fig.savefig('conf_matrix.jpg')


# Cross Validation

# In[36]:


from sklearn.model_selection import cross_val_score, cross_validate
accuracies = cross_validate(estimator = classifier, X = X_train, y = y_train, cv = 10,scoring=('f1_macro','accuracy'))
print(accuracies['test_accuracy'].mean())
print(accuracies['test_f1_macro'].mean())
svm_acc = accuracies['test_accuracy']


# In[37]:


import scikitplot as skplt
pl = skplt.estimators.plot_learning_curve(classifier, X_train, y_train,
                                     cv=10, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Learning Curve");

plot_cross = pl.get_figure()
plot_cross.savefig('svm_cross.jpg')


# In[41]:


# boxplot algorithm comparison
results = [rand_acc, desc_acc, svm_acc]
names = ['Random_Forest', 'Decision_Tree', 'Support Vector Machine']
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
fig.savefig('algo_comp.jpg')


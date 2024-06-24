#!/usr/bin/env python
# coding: utf-8

# ### Task 0
# Import numpy and pandas

# In[1]:


import numpy as np
import pandas as pd


# ### Task 1
# Read the Google Playstore csv file and create a pandas dataframe object. Review the first 5 rows

# In[2]:


df = pd.read_csv('googleplaystore.csv')
df.head()


# In[ ]:





# ### Task 2
# Identify the column(s) which contain missing values. Count the number of missing values for each column

# In[7]:


df.isnull().sum()


# In[ ]:





# ### Task 3
# How many different categories of the apps the dataset contains

# In[8]:


df['Category'].nunique()


# In[ ]:





# ### Task 4
# Draw a boxplot of the dataset

# In[9]:


df.boxplot(column=['Rating'])


# In[ ]:





# ### Task 5
# You have noticed an outlier for the rating values from the box plot above. Identify the row which contains that outlier

# In[10]:


df[df['Rating']>5.0]


# In[ ]:





# ### Task 6
# Please fill the missing values in the `Rating` column with the average rating from that column

# In[11]:


df['Rating'].fillna(value=df['Rating'].mean(), inplace = True)
df.isnull().sum()


# In[ ]:





# ### Task 7
# Identify the exact data type for values in the `Last Updated` column

# In[12]:


type(df['Last Updated'][0])


# In[ ]:





# ### Task 8
# Convert the values in the `Last Updated` column to Timestamp type. You will find an error. Identify the row which causes the error

# In[13]:


notnull = df['Last Updated'].notnull()
not_datetime = pd.to_datetime(df['Last Updated'], errors='coerce').isna()
not_datetime = not_datetime & notnull
df[not_datetime]


# In[ ]:





# ### Task 9
# Remove the row you identified from the previous task

# In[14]:


df = df.drop(df[not_datetime].index)


# In[ ]:





# ### Task 10
# Convert the values in the Last Updated column to Timestamp type and display the `Last Updated` column

# In[15]:


df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df['Last Updated']


# In[ ]:





# ### Task 11
# Convert the `Last Updated` column into the row index of the dataframe

# In[16]:


df.set_index('Last Updated', inplace=True)
df.head()


# In[ ]:





# ### Task 12
# Sort the dataset index in ascenfing order and display the first five rows of the dataframe

# In[17]:


df = df.sort_index()
df.head()


# In[ ]:





# ### Task 13
# What is the average rating for all apps created in year 2018

# In[18]:


df.loc["2018"]["Rating"].mean()


# In[ ]:





# ### Task 14
# What are the total different number of the categories for all apps

# In[19]:


df["Category"].nunique()


# In[ ]:





# ### Task 15
# What are the top 5 categories of the apps?

# In[20]:


df["Category"].value_counts()[:5]


# In[ ]:





# ### Task 16
# Identify the exact data type for values in the `Reviews` column

# In[21]:


type(df["Reviews"][0])


# In[ ]:





# ### Task 17
# Convert the values in the `Reviews` into integer and display your result for verification

# In[22]:


df["Reviews"] = df["Reviews"].astype(np.int64)
type(df["Reviews"][0])


# In[ ]:





# ### Task 18
# Display the list of all possible price values from the `Price` column. Identify the exact datatype of the values in the `Price` column. Convert the values in that coulumn to float (hint, you need to first remove the $ sign)

# In[23]:


df['Price'].unique()


# In[ ]:





# ### Task 19
# Identify the exact data type for values in the `Price` column

# In[24]:


type(df['Price'][0])


# In[ ]:





# ### Task 20
# Replace the `$` sign in front of the price. Convert the price into floating point numbers

# In[25]:


df['Price'] = df.Price.apply(lambda x: x.strip('$'))
df['Price'] = pd.to_numeric(df['Price'])
df['Price'].dtypes


# In[ ]:





# ### Task 21
# Find the apps with a price range between `$100` and `$200`

# In[26]:


df[df['Price'].between(100, 200, inclusive=False)]


# In[ ]:





# ### Task 22
# Identify the exact data type for values in the `Installs` column. After that, convert the values into integer. Hint, you need to remove the `+` and `,` signs from the number first before the convertion. Display the first five rows of the dataframe

# In[27]:


type(df['Installs'][0])
df['Installs']=df['Installs'].str.replace(',', '')
df['Installs']=df['Installs'].str.replace('+', '')
df['Installs']=df['Installs'].astype(int)
df.head()


# In[ ]:





# ### Task 23
# Fine the average `Rating`, `Reviews`, `Installs` and `Price` for each year

# In[28]:


df.groupby(pd.Grouper(level='Last Updated',freq = 'Y')).mean()


# In[ ]:





# ### Task 24
# Observe the values in the `Genres` column. You may notice that some values contain multiple genres which are separated by `;`. Create a new coulum named `Genre` and each value of the new column contains a list of genre names. Display the first five rows of the new dataframe

# In[29]:


df['Genre']=df['Genres'].str.split(';')
df.head()


# In[ ]:





# ### Task 25
# If an element in the `genre` column has multiple values in the list, separate each genre name into different rows. Display the last five rows of the new dataframe. Hint, research the pandas `explode` function

# In[30]:


df = df.explode('Genre')
df.tail()


# In[ ]:





# ### Task 26
# Draw a bar plot to display the number of apps for each genre. Which genre has the most apps?

# In[31]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,5))
df['Genre'].value_counts().sort_values(ascending = False).plot.bar()


# In[ ]:





# ### Part II: Wine Prediction Using KNN
# In this part of the final project, you will use the Wine dataset provided by sklearn to perform the wine classification

# ### Task 26
# Load the wine dataset from sklearn (use `load_wine` from the `sklearn.datasets` module) and conduct the necessary processing on the training dataset

# In[32]:


from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine = load_wine(return_X_y=False)

mm = StandardScaler()
result = mm.fit_transform(wine.data)
result


# In[ ]:





# ### Task 27
# Split the dataset into training and test sets with a test size of 25%. Use `random_state=66` for reproducibility. Set `K=1` for KNN and generate the classification report

# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


Xtrain, Xtest, ytrain, ytest = train_test_split(result, wine.target, test_size = 0.25, random_state = 66)


knn = KNeighborsClassifier(n_neighbors=1)

# fit the k-nearest neighbors classifier from the training dataset
knn.fit(Xtrain,ytrain)

# predict the class labels for the provided data
pred = knn.predict(Xtest)

# return the mean accuracy on the given test data and labels
knn.score(Xtest, ytest)

print(classification_report(ytest, pred))


# In[ ]:





# ### Task 28
# Using the Exhaustive Grid Search to find the best `K` in the `range(1, 20, 2)`. Using `10-fold`

# In[34]:


from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
param = {'n_neighbors': range(1, 20, 2)}
gc = GridSearchCV(knn, param_grid=param, cv=10)
gc


# In[ ]:





# ### Task 29
# What is the best score from Task 28

# In[35]:


gc.fit(Xtrain, ytrain)
gc.score(Xtest, ytest)
gc.best_score_


# In[ ]:





# ### Task 30
# What is the best estimator from Task 28

# In[36]:


gc.best_estimator_


# In[ ]:





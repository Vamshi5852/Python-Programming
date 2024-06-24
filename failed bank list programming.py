#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:






# ### Task 0 (0 pt)
# Run the following cell

# In[92]:


import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# ### Task 1
# **Import numpy as np** <br> **Import pandas as pd**

# In[86]:


import numpy as np
import pandas as pd


# ### Task 2 
# Read the failed bank list from FDIC website at https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/ <br>
# Display the first five rows of the table. You must use the `read_html` function to complete this task    

# In[93]:


df = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')
df[0].head()


# In[3]:





# ### Task 3
# Display the Column names of the table

# In[94]:


df[0].columns


# In[4]:





# ### Task 4
# Check to see whether there is any empty values in each column

# In[95]:


df[0].isna().any()


# In[5]:





# ### Task 5
# Check to see whether there is any duplicated row

# In[96]:


df[0].duplicated().sum()


# In[6]:





# ### Task 6
# 
# You may notice that the column names of the table is not correct. Please change some of the column names accoording to the following list. <br>
# `Bank NameBank` -> `Bank Name` <br>
# `CityCity` -> `City` <br>
# `StateSt` -> `State` <br>
# `Closing DateClosing` -> `Closing Date`

# In[97]:


df[0] = df[0].rename(columns={'Bank NameBank': 'Bank Name', 'CityCity': 'City','StateSt':'State','Closing DateClosing':'Closing Date'})
df[0].head()


# In[7]:





# ### Task 7
# How many failed banks were based in Missouri?

# In[99]:


len(df[0][df[0].State =='MO'])


# In[8]:





# ### Task 8
# Which city has the most numbers of failed banks?

# In[100]:


df[0]['City'].value_counts().index[0]


# In[9]:





# ### Task 9
# Verify the date in the 'Closing Date' column is a String not the DateTime object

# In[101]:


type(df[0]['Closing Date'][0])


# In[10]:





# ### Task 10
# Which year has the most failed banks?

# In[74]:


year_df=pd.DataFrame(i.year for i in pd.to_datetime(df[0]['Closing Date']))
str(year_df.value_counts().index[0][0])


# In[13]:





# ### Task 11
# Which month has the most failed banks?

# In[75]:


month_df=pd.DataFrame(i.month_name() for i in pd.to_datetime(df[0]['Closing Date']))
str(month_df.value_counts().index[0][0])


# In[14]:





# ### Task 12
# Which institution acquired the second most numbers of failed banks?

# In[76]:


df[0]['Acquiring InstitutionAI'].value_counts().index[1]


# In[15]:





# ### Task 13
# Change the date in the `Closing Date` column to the dateTime object and display the first five rows of the table.

# In[77]:


df[0]['Closing Date']=pd.to_datetime(df[0]['Closing Date'])
df[0].head()


# In[16]:





# ### Task 14
# Display the Bank information which has a cert number 33901

# In[78]:


df[0][df[0]['CertCert']==33901]


# In[17]:





# ### Task 15
# How many banks failed between January 1, 2008 and December 31, 2010?

# In[79]:


len(df[0][df[0]['Closing Date'].between('2008-01-01', '2010-12-31')])


# In[18]:





# ### Task 16
# Reorganize the table to make the State and City as the indexes of the table. Display the first 20 rows of the new table

# In[102]:


df[0]=df[0].set_index(['State', 'City']).sort_index()
df[0].head(20)


# In[ ]:





# ### Task 17
# Display failed banks based in Kansas City, MO

# In[103]:


df[0].loc[('MO','Kansas City')]


# In[20]:





# ### Task 18
# Display failed banks in Missouri and Kansas

# In[82]:


df[0].loc[['MO','KS']]


# In[21]:





# ### Task 19
# Display failed banks based in the Kansas City metro area which include Kansas City, Overland Park, Leawood and Olathe

# In[83]:


df[0].loc[[('MO','Kansas City'),('KS','Overland Park'),('KS','Leawood'),('KS','Olathe')]]


# In[22]:





# ### Task 20
# How many failed banks' names include the word `National`?

# In[84]:


len(df[0][df[0]['Bank Name'].str.contains('National')])


# In[23]:





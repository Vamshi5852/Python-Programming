#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Making the Default changes
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Loads the ss13hil.csv file
# reading the data

df = pd.read_csv("ss13hil.csv")
# Task
# Create 3 tables:
# TABLE 1: Statistics of HINCP - Household income (past 12 months), grouped by HHT - Household/family type

# Table 1 - Descriptive Statistics of HINCP, grouped by HHT

HHT_types = { 1 : 'Married couple household',
                 2 : 'Other family household:Male householder, no wife present',
                 3 : 'Other family household:Female householder, no husband present',
                 4 : 'Nonfamily household:Male householder:Living alone',
                 5 : 'Nonfamily household:Male householder:Not living alone',
                 6 : 'Nonfamily household:Female householder:Living alone',
                 7 : 'Nonfamily household:Female householder:Not living alone' }

htt_index_df = pd.DataFrame(HHT_types.items())
htt_index_df.index.name = 'HHT - Household/family type'
hht = df.groupby('HHT').aggregate({'HINCP':['mean','std','count','min','max']})
hht.columns = hht.columns.levels[1]
hht.index = hht.index.astype(int)
hht = hht.merge(htt_index_df, left_index=True, right_on=[0])
hht.set_index(1, inplace=True)
hht = hht.drop(0,axis=1)
hht.index.name = None
hht['min'] = hht['min'].astype(int)
hht['max'] = hht['max'].astype(int)
hht.sort_values(by='mean', inplace=True, ascending=False)
new_hht = hht


# Table 2
HHL_types ={ 1: 'English only', 
                   2: 'Spanish', 
                   3: 'Other Indo-European languages',
                   4: 'Asian and Pacific Island languages', 
                   5:'Other language'} 
    
acess_values = { 1:'Yes w/ Subsrc.', 2:'Yes, wo/ Subsrc.', 3:'No'}
HLL = df[['HHL','ACCESS','WGTP']].dropna(subset=['WGTP','ACCESS','HHL'])
wgtp_total = float(HLL.WGTP.sum())
hhl_sum = lambda x: "{0:.2f}".format(round(np.sum(x)/wgtp_total* 100,2)) + '%'
hhl_access = HLL.pivot_table(['WGTP'],index='HHL',columns='ACCESS',aggfunc=hhl_sum,margins=True)
hhl_access.index.name = 'HHL - Household language'
hhl_access = hhl_access.rename(index=HHL_types,columns=acess_values)

# Table 3

desc = df.HINCP.describe(percentiles = [1/3.0, 2/3.0])
df['bin'] = pd.cut(df.HINCP,bins = [-np.Inf, desc['33.3%'], desc['66.7%'], np.Inf],labels = ['low', 'medium', 'high'])
df = df.groupby('bin').aggregate({'HINCP' : ['min', 'max', 'mean'],'WGTP' : ['sum']})
df.columns = ['min', 'max', 'mean', 'household_count']
df.index.name = 'HINCP'
df['min'] = df['min'].astype(int)
df['max'] = df['max'].astype(int)
HINCP=df


# Driver Code
if __name__ == '__main__':
    print('DATA-51100, Spring 2024\nNAME: JYOTHIKA JANUPALA\nPROGRAMMING ASSIGNMENT #7\n')
    print('*** Table 1 - Descriptive Statistics of HINCP, grouped by HHT ***')
    print(new_hht)
    print()
    print('*** Table 2 - HHL vs. ACCESS - Frequency Table ***')
    print(hhl_access)
    print('***Table 3 - Quantile Analysis of HINCP - Household income (past 12 months)***')
    print(HINCP)


# In[ ]:





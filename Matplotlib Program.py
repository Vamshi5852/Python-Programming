#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('\n')

# Import the required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

#Below function will return the dataframe of csv file
def read_csv_file(file_name):
    return pd.read_csv(file_name)
# Call above function
given_data = read_csv_file('ss13hil.csv')

print('Data Load Successfully! \n')
# make subplots funcion
def make_subplots(x,y,title):
    plot, ((UL, UR), (LL, LR)) = plt.subplots(x, y, figsize = (16,12))
    plot.suptitle(title,fontsize=16, fontweight="bold")
    return plot, UL , UR , LL , LR
# call above function
plots,upper_left,upper_right,lower_left,lower_right = make_subplots(2,2,'Required Output')


# Below Function is used to make the pie chart
def construct_pie_chart(df,position,y):
    HHL_df= df[['HHL']]
    HHL_count = (HHL_df.value_counts()).to_numpy()
    Col, texts = position.pie(HHL_count, startangle=-120.5)
    position.legend(Col, y, loc='upper left', bbox_to_anchor=(-0.23,1), frameon=True)
    position.set_title('Household Languages')
    position.set_ylabel('HHL')
print('Plots Are Constructing...\n')
print('Wait...............\n')
construct_pie_chart(given_data,upper_left,['English only','Spanish','Other Indo-European','Asian and Pacific Island languages','Other language'])

# Below function is used to construct the histogram
def construct_histogram(df,position):
    col_dataframe = df['HINCP']
    col_dataframe = pd.DataFrame(col_dataframe)
    col_dataframe = col_dataframe.assign(HINCP = col_dataframe['HINCP'].fillna(1))
    col_dataframe['HINCP'][col_dataframe['HINCP']<=1] = 1
    HINCP_values = col_dataframe['HINCP'].values
    Histbin = np.logspace(1,7,num=100)    
    position.hist(HINCP_values, bins=Histbin, facecolor='g', density='True', alpha=0.5, histtype='bar', range=(0, len(HINCP_values)))
    position.set_xscale('log')
    position.ticklabel_format(style='plain', axis='y')
    density=kde.gaussian_kde(df['HINCP'].dropna())
    position.plot(Histbin, density(Histbin), color='k', ls='dashed')
    position.set_title('Distribution of Household Income')
    position.set_ylabel('Density')
    position.set_xlabel('Household Income($)- Log Scaled')

construct_histogram(given_data,upper_right)

# Below function is used to construct the bar graph
def construct_bar_graph(df,position):
    vehicle_df = df['VEH']
    vehicle_df = vehicle_df.dropna()
    vehicle_count = (df.groupby('VEH')['WGTP'].sum()).to_numpy()
    y_axis = vehicle_count/1000
    x_axis = [i for i,x in enumerate(y_axis)]
    position.bar(x_axis, y_axis, color='red')
    position.set_title('Vehicles Available in Households')
    position.set_ylabel('Thousands of Households')
    position.set_xlabel('# of Vehicles')

construct_bar_graph(given_data,lower_left)


# Below function is used to construct the scatter plot graph
def construct_scatter_plot(df,position):
    df['TAXP'] = df['TAXP'].replace(1, 0)
    df['TAXP'] = df['TAXP'].replace(2, 1)
    for i in range(3,23):
        df['TAXP'] = df['TAXP'].replace(i, (i-2)*50)
    for i in range(23,63):
        df['TAXP'] = df['TAXP'].replace(i, (i-12)*100)
    df['TAXP'] = df['TAXP'].replace(63, 5500)
    for i in range(64,69):
        df['TAXP'] = df['TAXP'].replace(i, (i-58)*1000)   
    scatter_Plot = position.scatter(df['VALP'], df['TAXP'], marker='o', s=df['WGTP']/2, c=df['MRGP'], cmap='bwr', alpha=0.5)
    Bar_line=plt.colorbar(scatter_Plot)
    Bar_line.set_label('First Mortgage Payment (Monthly $)')
    position.set_xlim(0, 1200000)
    position.ticklabel_format(style='plain')
    position.set_title('Property Taxes vs Property Values')
    position.set_ylabel('Taxes($)', fontweight="bold")
    position.set_xlabel('Property Value($)')

construct_scatter_plot(given_data,lower_right)
print('All Plots Are Successfully Constructed!\n')
# save plots
plots.savefig('pums.png')

print('Plots Are Saved As pums.png file \n')


# In[ ]:





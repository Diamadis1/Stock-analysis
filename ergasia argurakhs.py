#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist


# # Shape of data set

# In[2]:


df=pd.read_csv("pricesperday1.csv")
df=df.dropna()
print('shape of data'),df.shape
df


# # Plotting all stocks

# In[28]:


# Timeseries for the first stock
# df['Stock1'].plot(figsize=(12,5))
df.plot()
plt.legend(ncol=7, loc="upper left")
plt.rcParams["figure.figsize"] = (15,10) 


# # 1. Plotting logarithmic returns of stocks
# ![log_returns.png](attachment:log_returns.png)

# In[31]:


# Log Returns plots
plt.rcParams["figure.figsize"] = (6.4, 4.8)    
for i in range (1,21):
    temp=df['Stock'+str(i)]
    df['Stock'+str(i)]
    k=[]

    for j in range(1,len(temp)):
        k.append(temp[j]-temp[j-1])
        font1 = {'family':'serif','color':'blue','size':20}
    plt.figure()
    plt.title('Stock'+str(i), fontdict = font1) 
    plt.plot(k,c='black')
    plt.axhline(y=0, c='k')


# In[32]:


# LogReturns for dataset
returns = pd.DataFrame()
for i in df:
    returns[i] = np.log(df[i]).diff()
returns = returns[1:]
    


# In[37]:


# First 5 rows to check 
returns.head()


# In[38]:


# Some usefull info
returns.describe()


# In[39]:


# Correlation on log returns for distance matrix
returns.corr()


# # 2. Plotting correlation matrix for the data set 
# ![2022-07-14.png](attachment:2022-07-14.png)

# In the 2nd part of the exercise, the correlations between the stocks had to be calculated. These correlations are shown as a heatmap. We can use this heatmap to easily distinguish some elements between the stocks, such as e.g. that Stock10 and Stock19 show a very reduced correlation compared to the rest, as can be seen from their dark color. At the same time, the first 8 stocks seem to show an increased correlation between them, especially Stocks5-7.

# In[40]:


# Dataset for crosscorrelation matrix
corrMatrix = returns.corr()


# In[57]:


#Correlation matrix of dataset 
fig, ax = plt.subplots(figsize=(15,15))
# sn.heatmap(corrMatrix, annot=True)
cmap = sn.dark_palette('#3fdd01',as_cmap=True)
map = sn.heatmap(corrMatrix,annot=True,cmap=cmap)
plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.title("Correlation matrix of stocks")
plt.show()


# # 3. Plotting distance matrix for the data set
# ![distance_matrix.png](attachment:distance_matrix.png)

# In the third part of the exercise, we calculate the distance between the shares which can be seen below. As expected the matrix seems to be the exact opposite to the correlations matrix. The values ranges from 0 to almost 2 depending on the distance the shares have to each other.

# In[119]:


A=squareform(pdist(corrMatrix.loc[['Stock1','Stock2','Stock3','Stock4','Stock5','Stock6','Stock7','Stock8','Stock9',
                                 'Stock10','Stock11','Stock12','Stock13','Stock14','Stock15','Stock16','Stock17',
                                 'Stock18',
                                'Stock19','Stock20']]))
fig, ax = plt.subplots(figsize=(15,15))
cmap = sn.dark_palette('#3fdd01',as_cmap=True)
map = sn.heatmap(A,annot=True,cmap=cmap)
plt.yticks(rotation=360)
# plt.xticks(rotation=45)
plt.title("Distance matrix of stocks")
plt.show()


# In[ ]:





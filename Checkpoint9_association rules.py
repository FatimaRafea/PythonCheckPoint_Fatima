#!/usr/bin/env python
# coding: utf-8

# Association Rules
# Let's say you are a Machine Learning engineer working for a clothing company and you want to adopt new strategies to improve the company's profit.
# 
# Use this dataset and the association rules mining to find new marketing plans. 
# 
# Note here that one of the strategies can be based on which items should be put together
# 
# dataset = [['Skirt', 'Sneakers', 'Scarf', 'Pants', 'Hat'],
# 
#     ['Sunglasses', 'Skirt', 'Sneakers', 'Pants', 'Hat'],
# 
#     ['Dress', 'Sandals', 'Scarf', 'Pants', 'Heels'],
# 
#     ['Dress', 'Necklace', 'Earrings', 'Scarf', 'Hat', 'Heels', 'Hat'],
# 
#    ['Earrings', 'Skirt', 'Skirt', 'Scarf', 'Shirt', 'Pants']]
# 
# Bonus: try to do some visualization before applying the Apriori algorithm.

# In[33]:


dataset = [['Skirt', 'Sneakers', 'Scarf', 'Pants', 'Hat'],

    ['Sunglasses', 'Skirt', 'Sneakers', 'Pants', 'Hat'],

    ['Dress', 'Sandals', 'Scarf', 'Pants', 'Heels'],

    ['Dress', 'Necklace', 'Earrings', 'Scarf', 'Hat', 'Heels', 'Hat'],

   ['Earrings', 'Skirt', 'Skirt', 'Scarf', 'Shirt', 'Pants']]


# In[34]:


import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
te=TransactionEncoder()
te_ary=te.fit(dataset).transform(dataset)    #Apply one-hot-encoding on our dataset
df1=pd.DataFrame(te_ary, columns=te.columns_)  #Creating a new DataFrame from our Numpy array
df1


# In[35]:


from mlxtend.frequent_patterns import apriori
apriori(df1, min_support=0.6)


# In[36]:


frequent_itemsets=apriori(df1, min_support=0.6, use_colnames=True) #Instead of column indices we can use column names.
frequent_itemsets


# In[37]:


from mlxtend.frequent_patterns import association_rules 
association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7) # associate itemsets with confidence over 70%.


# In[38]:


from mlxtend.frequent_patterns import association_rules 
association_rules(frequent_itemsets,metric="lift",min_threshold=1.25)


# Whoever buys a skirt, will most likely buy pants too. 
# 
# Let's try it with the big dataset now. 

# In[39]:


# for basic operations
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns


# for market basket analysis
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[40]:


df = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
df


# In[41]:


#checking for NaN values 
df.isnull().sum().sum()


# In[42]:


#checking for NaN values in each feature 

df.isnull().sum()


# In[43]:


# looking at the frequency of most popular items 

plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
df[0].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.show()


# In[44]:


# making each customers shopping items an identical list
trans = []
for i in range(0, 7501):
    trans.append([str(df.values[i,j]) for j in range(0, 20)])

# conveting it into an numpy array
trans = np.array(trans)

# checking the shape of the array
print(trans.shape)


# In[45]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
data = te.fit_transform(trans)
data = pd.DataFrame(data, columns = te.columns_)

# getting the shape of the data
data.shape


# In[46]:


import warnings
warnings.filterwarnings('ignore')

# getting correlations for 121 items would be messy 
# so let's reduce the items from 121 to 50

data = data.loc[:, ['mineral water', 'burgers', 'turkey', 'chocolate', 'frozen vegetables', 'spaghetti',
                    'shrimp', 'grated cheese', 'eggs', 'cookies', 'french fries', 'herb & pepper', 'ground beef',
                    'tomatoes', 'milk', 'escalope', 'fresh tuna', 'red wine', 'ham', 'cake', 'green tea',
                    'whole wheat pasta', 'pancakes', 'soup', 'muffins', 'energy bar', 'olive oil', 'champagne', 
                    'avocado', 'pepper', 'butter', 'parmesan cheese', 'whole wheat rice', 'low fat yogurt', 
                    'chicken', 'vegetables mix', 'pickles', 'meatballs', 'frozen smoothie', 'yogurt cake']]

# checking the shape
data.shape


# In[47]:


data.head()


# In[48]:


import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
te=TransactionEncoder()
te_ary=te.fit(dataset).transform(dataset)    #Apply one-hot-encoding on our dataset
df=pd.DataFrame(te_ary, columns=te.columns_)  #Creating a new DataFrame from our Numpy array
df


# In[49]:


from mlxtend.frequent_patterns import apriori
apriori(df, min_support=0.6)


# In[53]:


#Support
frequent_itemsets=apriori(data, min_support=0.05, use_colnames=True)
frequent_itemsets


# In[54]:


#confidence 
association_rules(frequent_itemsets,metric="confidence",min_threshold=0.1)


# In[56]:


#lift
table_data = association_rules(frequent_itemsets,metric="lift",min_threshold=1)
table_data


# Mineral water is obviously the main item which is recurrent. People who buy mineral water (antecedents), end up buying other items too (consequents) One strategy that can be followed is selling mineral water next to other items or having multiple mineral water shelves distributed in different parts of the market. 

#!/usr/bin/env python
# coding: utf-8

# ### Comparison Operators as UFuncs
# We have learned that using `+`, `-`, `*`, `/`, and others on arrays leads to element-wise operations. NumPy also implements comparison operators such as `<` (less than) and `>` (greater than) as element-wise ufuncs. The result of these comparison operators is always an array with a Boolean data type. All six of the standard comparison operations (`<`, `<=`, `>`, `>=`, `==` and `!=`) are available
# 
# <div>
# <img src="attachment:f1.png" width="350"/>
# </div>
# 
# As in the case of arithmetic operators, the comparison operators are implemented as ufuncs in NumPy; for example, when you write `x < 3`, internally NumPy uses `np.less(x, 3)`

# In[1]:


import numpy as np
x = np.array([1, 2, 3, 4, 5])
x < 3  # less than


# In[2]:


x == 3  # equal


# It is also possible to do an element-wise comparison of two arrays, and to include compound expressions. Comparisons between arrays of the same size yield boolean arrays

# In[3]:


(2 * x)


# In[5]:


(x ** 2)


# In[4]:


(2 * x) == (x ** 2)


# Just as in the case of arithmetic ufuncs, these will work on arrays of any size and shape. In each case, the result is a Boolean array

# In[ ]:


# Two dimensional example
rng = np.random.RandomState(0)
x = rng.randint(10, size = (3, 4))
x


# In[ ]:


x < 6


# ### Working with Boolean Arrays

# #### Counting Entries
# To count the number of `True` entries in a Boolean array, `np.count_nonzero` is useful. In Python, all nonzero integers will be evaluated as `True`.

# In[6]:


x


# In[8]:


# how many values less than 6?
np.count_nonzero(x < 6)


# Another way to get at this information is to use `np.sum`; in this case, `False` is interpreted as 0, and `True` is interpreted as 1. The benefit of `sum()` is that like with other NumPy aggregation functions, this summation can be done along rows or columns as well.

# In[9]:


np.sum(x < 6)


# In[13]:


# how many values less than 6 in each row?
np.sum(x < 6)


# In[7]:


np.sum(x < 6, axis = 0)


# If we're interested in quickly checking whether any or all the values are true, we can use `np.any` or `np.all`. `np.all` and `np.any` can be used along particular axes as well.

# In[11]:


# are there any values greater than 8?
np.any(x > 8)


# In[12]:


# are there any values less than zero?
np.any(x < 0)


# In[14]:


# are all values less than 10?
np.all(x < 10)


# In[15]:


# are all values equal to 6?
np.all(x == 6)


# In[16]:


# are all values in each row less than 8?
np.all(x < 8, axis = 1)


# ### Boolean Operators
# Python's bitwise logic operators, `&`, `|`, `^`, and `~`. Like with the standard arithmetic operators, NumPy overloads these as ufuncs which work element-wise on (usually Boolean) arrays. Combining comparison operators and Boolean operators on arrays can lead to a wide range of efficient logical operations.
# 
# <div>
# <img src="attachment:f1.png" width="350"/>
# </div>

# ### Case Study: Counting Rainy Days

# In[20]:


import pandas as pd

rainfall = pd.read_csv('Seattle2014.csv')
rainfall.head()


# In[21]:


# use pandas to extract rainfall inches as a NumPy array
rainfall = pd.read_csv('Seattle2014.csv')['PRCP'].values
inches = rainfall / 254.0  # 1/10mm -> inches
inches.shape


# In[22]:


# Number of days with rainfall between 0.5 and 1.0 inches
np.sum((inches > 0.5) & (inches < 1))


# Note that the parentheses here are important - because of operator precedence rules. With parentheses removed this expression would be evaluated as follows, which results in an error.
# 
# `inches > (0.5 & inches) < 1`

# In[23]:


print("Number days without rain:      ", np.sum(inches == 0))
print("Number days with rain:         ", np.sum(inches != 0))
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.2 inches  :", np.sum((inches > 0) & (inches < 0.2)))


# ### Boolean Arrays as Masks
# In the preceding section we looked at aggregates computed directly on Boolean arrays. A more powerful pattern is to use Boolean arrays as masks, to select particular subsets of the data themselves. 

# In[34]:


rng = np.random.RandomState(0)
x = rng.randint(10, size = (3, 4))
x


# In[35]:


x < 5


# In[36]:


np.sum(x < 5)


# In[37]:


x[x < 5]


# What is returned is a one-dimensional array filled with all the values that meet this condition; in other words, all the values in positions at which the mask array is `True`.

# In[38]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names


# In[39]:


data = np.random.randn(7, 4)
data


# Suppose each name corresponds to a row in the data array and we wanted to select all the rows with corresponding name Bob. Like arithmetic operations, comparison with array are also vectorized.

# In[40]:


names == 'Bob'


# In[41]:


data[names == 'Bob']


# The boolean array must be of the same length as the array axis it is indexing. You can even mix and match boolean arrays with slices, indices (integers) or fancy indices (sequences of integers) which will be covered later

# In[42]:


data[names == 'Bob', 2:]


# In[43]:


data[names == 'Bob', 3]


# In[45]:


# revisit this example after we cover fancy indexing
data[names == 'Bob', [3, 0]]


# To select everything but Bob, you can either use `!=` or negate the condition using `~`

# In[44]:


names != 'Bob'


# In[ ]:


data[names != 'Bob']


# In[ ]:


data[~(names == 'Bob')]


# Selecting two of the three names to combine multiple boolean conditions.

# In[ ]:


data[(names == 'Bob') | (names == 'Will')]


# **Selecting data from an array by boolean indexing always creates a copy of the data, even if the returned array is unchanged**. Also keep in mind that the Python keywords `and` and `or` do not work with boolean arrays. Use `&` and `|` instead. See the note below 

# ### A Note
# One common point of confusion is the difference between the keywords `and` and `or` on one hand, and the operators `&` and `|` on the other hand. When would you use one versus the other?
# 
# The difference is this: `and` and `or` gauge the truth or falsehood of one element, while `&` and `|` refer to elemet-wise operation.

# In[46]:


A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B


# In[48]:


A and B


# In[49]:


matrix = np.arange(36).reshape(3, 12)
matrix


# In[50]:


rows_on = np.array([True, False, True])
cols_on = np.array([False, True, False] * 4)


# In[51]:


matrix[rows_on]


# In[52]:


matrix[rows_on, cols_on] 


# In[ ]:


#[True, False], [True, True], [True, False] ...
#[False, False], [False, True], [False, False] ...
#[True, False], [True, True], [True, False] ...
matrix[np.ix_(rows_on, cols_on)] # using boolean indexing


# Using `np.ix_` one can quickly construct index arrays that will index the cross product. `a[np.ix_([1,3],[2,5])]` returns the array `[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]`

# In[ ]:


matrix[[0, 2], 1::3] # same effect using the regular indexing


# Setting values with boolean arrays works in a common sense way. To set all of the negative values in data to 0, we only need to do

# In[ ]:


data[data < 0] = 0
data


# Setting whole rows or columns using a one-dimensional boolean array is also easy

# In[ ]:


data[names != 'Joe'] = 7
data


# In[ ]:


# construct a mask of all rainy days
rainy = (inches > 0)

# construct a mask of all summer days (June 21st is the 172nd day)
days = np.arange(365)
summer = (days > 172) & (days < 262)

print("Median precip on rainy days in 2014 (inches):   ",
      np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches):  ",
      np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ",
      np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):",
      np.median(inches[rainy & ~summer]))


# In[ ]:


inches[rainy]


# ### Fancy Indexing
# In the previous sections, we saw how to access and modify portions of arrays using simple indices (e.g., `arr[0]`), slices (e.g., `arr[:5]`), and Boolean masks (e.g., `arr[arr > 0]`). In this section, we'll look at another style of array indexing, known as fancy indexing. **It is a term adopted by NumPy to describe indexing using integer arrays**. Fancy indexing is like the simple indexing we've already seen, but we pass arrays of indices in place of single scalars. This allows us to very quickly access and modify complicated subsets of an array's values. Fancy indexing is conceptually simple: it means passing an array of indices to access multiple array elements at once.
# 
# The data generation functions in `np.random` use a global random seed. To avoid global state, you can use `np.RandomState` to create a random number generator isolated from others

# In[53]:


rng = np.random.RandomState(42)

x = rng.randint(100, size = 10)
x


# In[54]:


# access three different elements
[x[3], x[7], x[2]]


# In[55]:


# using fancy indexing
x[[3, 7, 2]]


# When using fancy indexing, the shape of the result reflects the shape of the index arrays rather than the shape of the array being indexed

# In[56]:


ind = np.array([[3, 7],
                [4, 5]])
x[ind]


# To select out a subset of the rows in a particular order, you can simply pass a list or ndarray of integers specifying the desired order

# In[59]:


X = np.arange(32).reshape((8, 4))
X


# In[60]:


X[4]


# In[61]:


X[[4, 3, 0, 6]]


# Using negative indices selects rows from the end

# In[62]:


X[[-3, -5, -7]]


# Fancy indexing also works in multiple dimensions. Like with standard indexing, the first index refers to the row, and the second to the column. Passing multiple index arrays does something slightly different. **It selects a one-dimensional array of elements corresponding to each tuple of indices**. Regardless of how many dimensions the array has (here, only 2), the result of fancy indexing with multiple integer arrays is always one-dimensional.

# In[58]:


X = np.arange(12).reshape((3, 4))
X


# In[68]:


# two index arrays with the same shape
row = np.array([0, 1, 2])

col = np.array([2, 1, 3])
X[row, col]


# In[ ]:


X[row]


# Notice that the first value in the result is `X[0, 2]`, the second is `X[1, 1]`, and the third is `X[2, 3]`. The behavior of fancy indexing in the above case is a bit different from what you might have expected. To get the rectangular region formed by selecting a subset of the matrix's rows and columns, see the example below

# In[69]:


X[[0, 1, 2]][:, [2, 1, 3]]


# In[70]:


X[:, [2, 1, 3]]


# In[71]:


X[[0, 2], 1:3]


# The pairing of indices in fancy indexing follows all the broadcasting rules. If we combine a column vector and a row vector within the indices, we get a two-dimensional result

# In[72]:


# two index arrays with different shape
#       2        1        3

# 0   (0, 2)   (0, 1)   (0, 3)

# 1   (1, 2)   (1, 1)   (1, 3)

# 2   (2, 2)   (1, 2)   (2, 3)

X[row[:, np.newaxis], col]


# In[ ]:


X[np.ix_(row, col)]


# Here, each row value is matched with each column vector, exactly as we saw in broadcasting of arithmetic operations. It is always important to remember with fancy indexing that the return value reflects the broadcasted shape of the indices, rather than the shape of the array being indexed.

# In[ ]:


row[:, np.newaxis] * col


# ### Modifying Values with Fancy Indexing
# Just as fancy indexing can be used to access parts of an array, it can also be used to modify parts of an array

# In[ ]:


x = np.arange(10)
x


# In[ ]:


i = np.array([2, 1, 8, 4])
x[i] = 99
x


# We can use any assignment-type operator for this

# In[ ]:


x[i] -= 10
x


# Keep in mind that fancy indexing, unlike slicing, always copies the data into a new array. We can verify that using the following example

# In[ ]:


new_x_slice = x[i] # get a copy to new_x_slice
new_x_slice[:] = 66
print(new_x_slice)
x


# ### Combined Indexing
# Fancy indexing can be combined with the other indexing schemes we've seen.

# In[ ]:


X


# We can combine fancy and simple indices

# In[ ]:


X[2, [2, 0, 1]]


# We can also combine fancy indexing with slicing

# In[ ]:


X[1:, [2, 0, 1]]


# And we can combine fancy indexing with masking

# In[ ]:


mask = np.array([1, 0, 1, 0], dtype = bool)
X[row[:, np.newaxis], mask]


# ### Sorting Arrays
# To return a sorted version of the array without modifying the input, you can use `np.sort`

# In[74]:


# class function
x = np.array([2, 1, 4, 3, 5])
np.sort(x)


# In[75]:


x # the original array is not changed


# If you prefer to sort the array in-place, you can instead use the `sort` method of arrays

# In[76]:


# array method
x.sort()
x


# A related function is `argsort`, which instead returns the indices of the sorted elements

# In[77]:


x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
i


# The first element of this result gives the index of the smallest element, the second value gives the index of the second smallest, and so on. These indices can then be used (via fancy indexing) to construct the sorted array if desired

# In[78]:


x[i]


# #### Sorting Along Rows or Columns
# A useful feature of NumPy's sorting algorithms is the ability to sort along specific rows or columns of a multidimensional array using the `axis` argument. **Keep in mind that this treats each row or column as an independent array, and any relationships between the row or column values will be lost!**

# In[89]:


rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
X


# In[90]:


# sort each column of X
np.sort(X, axis = 0)


# In[91]:


# sort each row of X
np.sort(X, axis = 1)


# #### Partial Sorts: Partitioning
# Sometimes we're not interested in sorting the entire array, but simply want to find the `k` smallest values in the array. NumPy provides this in the np.partition function. `np.partition` takes an array and a number k; the result is a new array with the smallest `k` values to the left of the partition, and the remaining values to the right, in arbitrary order. **Within both partitions (left and right), the elements have arbitrary order.**

# In[97]:


x = np.array([7, 2, 100, 1, 6, 5, 4])
np.partition(x, 4)


# In[98]:


# if k is negative, the largest k number are put at the end of the array
x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, -3)


# Similarly to sorting, we can partition along an arbitrary axis of a multidimensional array

# In[99]:


X


# In[100]:


np.partition(X, 2, axis = 1)


# The result is an array where the first two slots in each row contain the smallest values from that row, with the remaining values filling the remaining slots. Finally, just as there is a np.argsort that computes indices of the sort, there is a `np.argpartition` that computes indices of the partition

# In[ ]:





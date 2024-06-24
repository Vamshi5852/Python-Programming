#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
np.random.seed(0)
values = np.random.randint(1, 10, size = 5)
values


# In[16]:


print(1.0 / values) # vectorized operation between scalar and array


# In[18]:


# vectorized operation between two arrays
a = np.arange(5)
b = np.arange(1, 6)
print(a)
print(b)


# In[19]:


a / b


# In[20]:


# vectorized operation on multidimensional arrays
x = np.arange(9).reshape((3, 3))
x


# In[21]:


2 ** x


# In[24]:


x ** 2


# Arithmetic operations with scalars propagate the scalar argument to each element in the array

# ### NumPy's UFuncs
# Ufuncs exist in two flavors: unary ufuncs, which operate on a single input, and binary ufuncs, which operate on two inputs.

# In[25]:


arr = np.arange(10)
arr


# In[26]:


# unary ufuncs
np.sqrt(arr)


# In[30]:


x = np.random.randn(8)
x


# In[31]:


y = np.random.randn(8)
y


# In[32]:


np.maximum(x, y)


# Here `np.maximum` computed the element-wise maximum of the elements in `x` and `y`

# While not common, a ufunc can return multiple arrays. `modf` is one example. It returns the fractional and integral parts of a floating point array

# In[36]:


arr = np.random.randn(7) * 5
arr


# In[37]:


remainder, whole_part = np.modf(arr)
remainder


# In[ ]:


whole_part


# Ufuncs accept an optional `out` argument that allows them to operate in-place on arrays

# In[ ]:


arr = np.abs(arr)
arr


# In[ ]:


np.sqrt(arr)


# In[ ]:


arr


# In[ ]:


np.sqrt(arr, out = arr)


# In[ ]:


arr


# #### Array Arithmetic
# NumPy's ufuncs feel very natural to use because they make use of Python's native arithmetic operators. The standard addition, subtraction, multiplication, division, remainder, exponentiation can all be used. In addition, these operations can be strung together however you wish, and the standard order of operations is respected. Each of these arithmetic operations are simply convenient wrappers around specific functions built into NumPy; for example, the `+` operator is a wrapper for the `add` function
# 
# <div>
# <img src="attachment:f1.png" width="350"/>
# </div>

# In[38]:


x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division
print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)


# In[39]:


-(0.5 * x + 1) ** 2


# In[40]:


np.add(x, 2)


# #### Other NumPy UFuncs: Absolute Values, Trig, Exponentials and Logrithms
# Just as NumPy understands Python's built-in arithmetic operators, it also understands Python's built-in absolute value function. The corresponding NumPy ufunc is `np.absolute`, which is also available under the alias `np.abs`. 

# In[41]:


x = np.array([-2, -1, 0, 1, 2])
abs(x)


# In[42]:


np.abs(x)


# In[46]:


theta = np.linspace(0, np.pi, 3)
print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))


# In[47]:


x = [-1, 0, 1]
print("x         = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))


# In[48]:


x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))


# In[49]:


x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))


# ### Expressing Conditional Logic as Array Operations
# The `np.where` function is a vectorized version of the ternary conditional expression `x if condition else y`. Suppose we have a boolean array and two arrays of values

# In[51]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# In[52]:


result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result


# In[53]:


result = np.where(cond, xarr, yarr)
result


# The second and third arguments to `np.where` do not need to be arrays. One or both of them can be scalars. A typical use of `np.where` in data analysis is to produce a new array of values based on another array. Suppose you have a matrix of randomly generated data and you want to replace all positive values with 3 and all negative values with -3

# In[54]:


arr = np.random.randn(4, 4)
arr


# In[55]:


arr > 0


# In[56]:


# it generates a copy
np.where(arr > 0, 3, -3)


# In[57]:


arr


# You can combine scalars and arrays when using `np.where`. For example, we will replace all positive values in arr with 3

# In[58]:


np.where(arr > 0, 3, arr) # set only positive values to 3


# In[59]:


arr


# ### Advanced UFunc Features

# #### Specifying Output
# For large calculations, it is sometimes useful to be able to specify the array where the result of the calculation will be stored. Rather than creating a temporary array, this can be used to write computation results directly to the memory location where you'd like them to be. For all ufuncs, this can be done using the `out` argument of the function

# In[60]:


x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out = y)
print(y)


# In[61]:


y = np.zeros(10)
np.power(2, x, out = y[::2])
print(y)


# #### Aggregations
# For binary ufuncs, there are some interesting aggregates that can be computed directly from the object.
# 
# <div>
# <img src="attachment:f1.png" width="350"/>
# </div>
# 
# For `min`, `max`, `sum`, and several other NumPy aggregates, a shorter syntax is to use methods of the array object itself. It is worthing noting that Python has built-in functions such as `sum`, `max` and `min`. They are not identical.
# 
# Most aggregates have a NaN-safe counterpart that computes the result while ignoring missing values, which are marked by the special IEEE floating-point `NaN` value
# 
# You should be aware that `NaN` is a bit like a data virus, it infects any other object it touches. Regardless of the operation, the result of arithmetic with `NaN` will be another `NaN`

# In[63]:


1 + np.nan


# In[64]:


0 *  np.nan


# In[65]:


vals2 = np.array([1, np.nan, 3, 4]) 


# In[66]:


vals2.sum()


# In[67]:


np.nansum(vals2)


# In[70]:


# Python's built-in max and min
big_array = np.random.rand(1000000)
min(big_array), max(big_array)


# In[71]:


np.min(big_array), np.max(big_array)


# In[74]:


get_ipython().run_line_magic('timeit', 'max(big_array)')
get_ipython().run_line_magic('timeit', 'np.max(big_array)')


# In[93]:


print(big_array.min(), big_array.max(), big_array.sum())


# In[ ]:


get_ipython().run_line_magic('timeit', 'sum(big_array)')
get_ipython().run_line_magic('timeit', 'np.sum(big_array)')


# #### Multi Dimensional Aggregates
# One common type of aggregation operation is an aggregate along a row or column. **By default, each NumPy aggregation function will return the aggregate over the entire array**. Aggregation functions take an additional argument specifying the axis along which the aggregate is computed. For example, we can find the minimum value within each column by specifying `axis = 0`. For aggregation along the row, use `axis = 1`. The `axis` keyword specifies the dimension of the array that will be collapsed, rather than the dimension that will be returned. So specifying `axis = 0` means that the first axis will be collapsed: for two-dimensional arrays, this means that values within each column will be aggregated

# In[78]:


M = np.random.random((3, 4))
M


# In[80]:


M.sum()


# In[81]:


M.sum(axis = 0)


# In[82]:


M.sum(axis = 1)


# In[88]:


M.min(axis = 0)


# In[87]:


M.max(axis = 1)


# In[103]:


M.argmax()


# In[104]:


M.argmin()


# In[105]:


M.argmax(axis = 1)


# In[106]:


M.argmax(axis = 0)


# In[108]:


M.cumsum()


# Functions like `min` and `max` take an optional `axis` argument that computes the statistic over the given axis, resulting in an array with **one fewer dimension**

# In[109]:


M


# In[110]:


M.cumsum(axis = 0)


# In[111]:


M.cumprod(axis = 1)


# In multidimensional arrays, accumulation functions such as `cumsum` return an array of the same size of the input array, but with the partial aggregates computed along with the indicated axis according to each lower dimensional slice

# ### `unique` and Other Set Logic
# NumPy has some basic set operations for one-dimensional ndarrays. A commonly used one is `np.unique` which returns the **sorted unique values** in an array

# In[112]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# In[113]:


ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# Contrast `np.unique` with the pure Python alternative

# In[114]:


sorted(set(names))


# Another function `np.in1d(x, y)` which computes a boolean array indicating whether each element of x is contained in y

# In[115]:


values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])


# #### Case Study: Average Heights of US Presidents

# In[100]:


import pandas as pd

import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # set plot style

get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


data = pd.read_csv('president_heights.csv')
heights = np.array(data['height(cm)'])
data.head()


# In[ ]:


heights


# In[ ]:


print("Mean height:       ", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height:    ", heights.min())
print("Maximum height:    ", heights.max())


# In[116]:


print("25th percentile:   ", np.percentile(heights, 25))
print("Median:            ", np.median(heights))
print("75th percentile:   ", np.percentile(heights, 75))


# In[ ]:


plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')


# ### Computation on Arrays: Broadcasting
# NumPy's universal functions can be used to vectorize operations and thereby remove slow Python loops. Another means of vectorizing operations is to use NumPy's broadcasting functionality. Broadcasting is simply a set of rules for applying binary ufuncs (e.g., addition, subtraction, multiplication, etc.) on arrays of different sizes.
# 
# For arrays of the same size, binary operations are performed on an element-by-element basis. Broadcasting allows these types of binary operations to be performed on arrays of different sizes. For example, we can just as easily add a scalar (think of it as a zero-dimensional array) to an array, add a one dimensioanl array to a two dimensional array or add two multi dimensional arrays with different dimensions.

# In[117]:


# add two arrays with same dimensions
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b


# In[118]:


# add an array with a scalar
a + 5


# In[119]:


c = np.ones((3, 3))
c


# In[120]:


# add a two dimensional array with a one dimensional array
# the one-dimensional array a is stretched, or broadcast across the second dimension in order to match the shape of c
c + a


# In[121]:


a = np.arange(3)
b = np.arange(3)[:, np.newaxis]

print(a)
print(b)


# In[122]:


a + b


# Just as before we stretched or broadcasted one value to match the shape of the other, here we've stretched both `a` and `b` to match a common shape, and the result is a two-dimensional array! The geometry of these examples is visualized in the following figure
# 
# <div>
# <img src="attachment:f1.png" width="350"/>
# </div>

# ### Rules of Broadcasting
# Broadcasting in NumPy follows a strict set of rules to determine the interaction between the two arrays:
# 
# - Rule 1: If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side.
# - Rule 2: If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
# - Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

# In[123]:


M = np.ones((3, 2))
a = np.arange(3)
M + a


# #### Centering an Array

# In[124]:


X = np.random.random((10, 3))
Xmean = X.mean(0)
Xmean


# In[130]:


X


# In[131]:


X_centered = X - Xmean


# In[132]:


X_centered.mean(0)


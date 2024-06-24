#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Mean = 0 
previousmean = None
Variance = 0 
number  = 0
n = 0
while number >= 0:
    print('')
    
    X_value = int(input('Enter number: '))
    if X_value < 0:
        number = -1
        break
    n += 1
    previousmean = Mean
    
    Mean += ((X_value - Mean) / n)
    if n == 1:	
        Variance = 0
    else:
        Variance = ((X_value - previousmean) ** 2 / n)+((n- 2) / (n- 1)) * Variance
    print('Mean is',Mean,' variance is' , Variance)


# In[ ]:





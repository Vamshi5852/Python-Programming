#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Loding the Required Modules
import numpy as np

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# writing the KNN class
class KNN:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        # return self.counter(k_neighbor_labels)
        return max(set(k_neighbor_labels),key=k_neighbor_labels.count)
    



# Driver code

# loding the training and testing data
train_data = np.array([file.strip().split(',') for file in open("./iris-training-data.csv")])
test_data =  np.array([file.strip().split(',') for file in open("./iris-testing-data.csv")])

# Extracting the data X_train,y_train,X_test,y_test
X_train,y_train,X_test,y_test = train_data[:,0:4], train_data[:,-1],test_data[:,0:4],test_data[:,-1]

# converting the data types of X_train,X_test to float32
X_train,X_test = X_train.astype(np.float32),X_test.astype(np.float32)


        

if __name__ == "__main__":
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)*100
        return round(accuracy,2)

    
    clf = KNN()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    # printing the output

    print("")
    print("#, True, Predicted")

    counter = 0

    [ print(f"{i+1},{y_test[i]},{predictions[i]}") for i in range(len(predictions))]
    print(f"Accuracy:{accuracy(y_test, predictions)}%")


# In[ ]:





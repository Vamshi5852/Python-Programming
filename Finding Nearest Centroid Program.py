#!/usr/bin/env python
# coding: utf-8

# In[2]:


#group data to its cluster value (nearest centroid)
def find_nearest_centroid_idx(centroids, point):

    latest_distance = float('inf')

    centroid_index = None

    for i, centroid in enumerate(centroids.values()):

        d = abs(centroid - point)

        if d < latest_distance:

            latest_distance = d

            centroid_index = i

    return centroid_index

 

#compute the mean of the data

def find_mean(numbers):   

    return sum(numbers) / len(numbers)

 

#compute the next centroids by finding the mean of each group

def find_centroids(clusters):

    centroids = {i: points[i] for i in range(len(clusters))}

    for i, cluster in enumerate(clusters.values()):

        centroids[i] = find_mean(cluster)

    return centroids

 

# =================================== Main Method =======================================

try:

    #read input file   

    input_file_name = 'prog2-input-data.txt'

    with open(input_file_name) as file:

        points = [float(line.rstrip()) for line in file]

    file.close()

       

    #promp user to enter the number and generate clusters
    print("")
    k = int(input("Enter the number of clusters: "))

    while k > len(points) or k <= 0:

        k = int(input("Please enter value between 1 to " + str(len(points)) + ": "))

        

    # Initialize variables

    centroids = {i: points[i] for i in range(k)}

    clusters = {i: [points[i]] for i in range(k)}

    previous_clusters = None

    point_assignments = {point : i for i, point in enumerate(points[:k])}

   

    # Repeat algorithm until points converge to clusters

    i = 0

    while True:

        previous_clusters = clusters

        clusters = {i: [] for i in range(k)}

   

        # Assign points to nearest cluster

        for point in points:

            nearest_centroid_idx = find_nearest_centroid_idx(centroids, point)

            clusters[nearest_centroid_idx].append(point)

            point_assignments[point] = nearest_centroid_idx

   

        # Update centroids based on new clusters

        updated_centroids = find_centroids(clusters)

        i = i + 1

   
        print("")
        print(f' Iteration {i}')

        [print(f'{key} {points}') for key, points in clusters.items()]

   

        if updated_centroids == centroids:

            break # kmeans has converged

        else:

            centroids = updated_centroids

    print('')

    [print(f'Point {point} in cluster {point_assignments[point]}') for point in points]

   

    # Write results to output file

    output_file = open("prog2-output-data.txt", "w")

    [output_file.write(f'Point {point} in cluster {point_assignments[point]} ') for point in points]

    output_file.close()

 

except FileNotFoundError:

    print(" No file found")  

except: 

    print(" Invalid input, needs to be a integer!!!")


# In[ ]:





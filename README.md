# SIH - Smart Travel - Team LinkedList

Repo contains python files corresponding to the ML Solutions that are a part of our project.

## Expected Occupancy at a Stop

Step 1: Used multiple linear regression to predict occupancy in the bus till it arrives at a stop based solely on the historical data (based on location and time factors)
Independent variables -> stop(location), month, week, day, time

Step 2: Used WLS to enhace the accuracy of the predicted occupancy as the data will vary greatly over the years

Step 3: Lambda Arch 
Uses batch processing and real-time processing to make a prediction about occupancy in the bus. Assigned weight = 0.7 to historical data and 0.3 to overall real-time data - 0.1 to each of the 3 variables


## Determining Safe path

It is the initial step for determining the safe paths. Uses kmeans to cluster regions according to their safety index (crime reports for various offences in different regions). Currently creating 5 clusters (used elbow method for determining number of clusters).

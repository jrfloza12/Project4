# The Correlation between Game Genre and Popularity
## Project4
<p align="center">
  <img src="https://github.com/jrfloza12/Project4/assets/122821004/cc0e6775-edd4-4b25-9779-1e8ccfbf2a73" alt="Screenshot">
</p>


The evolution of Video games shifted from just a past time to a culture that many young and old gravitate to. This opens an opportunity for many entrepenuers and companies looking the blue print for success in this particular industry. However, with literarlly hundreds of thousands of video games in the past 50 years how do you measure that?

Our project is not to re-invent the wheel or provide the answer to the blue print, but rather find the patterns that would lead to it. 
In our data set we want to include Names, Ratings, Genre and other features that could help analyze this further.

To start our project we needed to find a trustworthy set of information that can make our research feasible.

Instead of pulling from an API we opted to reasearch currently created DataSets since video games themselves are a popular topic in general. Out of all the options we had, Kaggle proved to contain what we needed so far.

Our story would begin with not how we compiled the data but rather what did we do with it.

Since we knew that the there would be a greater effort of time and coding just to predict the components of a successful video game, we decided on an unsupervised ML approach.
Like a certain article defines it as "The process of inferring underlying hidden patterns from historical data. Within such an approach, a machine learning model tries to find any similarities, differences, patterns, and structure in data by itself. No prior human intervention is needed. Picture a toddler. The child knows what the family cat looks like (provided they have one) but has no idea that there are a lot of other cats in the world that are all different. The thing is, if the kid sees another cat, he or she will still be able to recognize it as a cat through a set of features such as two ears, four legs, a tail, fur, whiskers, etc. In machine learning, this kind of prediction is called unsupervised learning. But when parents tell the child that the new animal is a cat – drumroll – that’s considered supervised learning." https://www.altexsoft.com/blog/unsupervised-machine-learning/.

Just like the example above, we are trying to identify the features and patterns that supervised learning would call out down the road if we opt to continue this process further.
This particular data set contains the imdb data with "Name", "year", "rating", "votes" and the "genre" categories.
There is about 20804 rows of data. 

## ETL

We used Pandas for the dataframe to clean up the data prior to clustering and submitting a model for machine learning.
From all the data we wanted to identify three unique features "genres, esrb rating and platforms.
We created three csv files from this set, [csv1](with_ratingvotes.csv) , [csv2](genreonly.csv) , [csv3](with_certificate.csv).

[ETL File](Cluster_ETL_2.ipynb)

## Kmeans 

[Kmeans](Cluster_DBSCAN.ipynb)

The next step was to cluster the data without defined categories. We want to find the groups in the data we have so far.
The functions that we used for this notebook was our go to "Pandas", "Numpy", "Sklearn". "StandardScaler", "Kmeans Cluster" and "Matplotlib" for visualization.

The csv file imported was the [csv1](with_ratingvotes.csv) since it had the most potential based off of the formatted data.
In the code we removed the names from the features and converted the boolean to integers.
We then normalized the data using standard scaler.

After normalizing we defined a data froma to hold the values for the number of clusters and the corresponding inertia.
This would then help with creating an elbow method for our file.

![Screenshot 2023-08-06 at 6 41 38 PM](https://github.com/jrfloza12/Project4/assets/122821004/9479becb-e97c-405c-876c-95dc2c560d9e)


We took the inertia values and used it in the Kmeans Cluster method.

![Screenshot 2023-08-06 at 6 43 24 PM](https://github.com/jrfloza12/Project4/assets/122821004/c2827fed-0605-4a3d-aaaa-5e94c8ca0299)

Once we got a look at this result we decided to remove "name" and "year" for another set. We were hoping that the curve would show a more prominent result.

![Screenshot 2023-08-06 at 6 53 26 PM](https://github.com/jrfloza12/Project4/assets/122821004/c7dd267b-3787-4fd2-af3a-ce6400a8fc06)
This was the result.

![Screenshot 2023-08-06 at 6 54 43 PM](https://github.com/jrfloza12/Project4/assets/122821004/f75b4605-3738-421c-b6ef-6eabca90c3f6)

With this dataframe cluster we were now able to create a Kmeans Cluster with scatter design.

![Screenshot 2023-08-06 at 6 56 23 PM](https://github.com/jrfloza12/Project4/assets/122821004/ac39da35-8359-4ba8-a8d3-83887728ad87)

Our next question was to find out how distributed were the ratings accross the clusters to see if there were any patterns there. 
The following chart was created to highlight this:

![Screenshot 2023-08-06 at 7 14 12 PM](https://github.com/jrfloza12/Project4/assets/122821004/8d2c6ae9-9123-4ddc-bf33-822dd6fcd9f0)

Cluster 4 was an outlier that caught our attention.
So we created a distribution on Votes across Clusters.

![Screenshot 2023-08-06 at 7 16 22 PM](https://github.com/jrfloza12/Project4/assets/122821004/359712e7-822b-4a2f-8fd9-65e5de2dc006)

Final piece of the cluster process was to see how much of the proportions of Genres are in Each Cluster.
We used a heatmap for this.

![Screenshot 2023-08-06 at 7 17 55 PM](https://github.com/jrfloza12/Project4/assets/122821004/5c50ff22-63ab-4fb5-baf8-45cbdc252360)



## DBSCAN

[DBSCAN](Cluster_DBSCAN.ipynb)

We opted for a DB scan to check for the density of the clustering. This will also help understand the outliers we originally saw in K means.
The same approach was used with [csv1](with_ratingvotes.csv) file and "Sklearn" for "DBSCAN".

We built a data fram and then selected only numerical features: "year", "rating", "votes".
Then we scaled the features.

We created a DBscan model and labeled the clusters.
This resulted in 93 clusters.

The next step was to add temporary labels to count the occurences.

![Screenshot 2023-08-06 at 8 00 01 PM](https://github.com/jrfloza12/Project4/assets/122821004/5c5ed4fe-6a52-4c4f-ab85-347922daa091)

### With the help of CHATGPT, it was suggested to consider the following notes:

Clustering is an unsupervised learning method, so the meaning and interpretation of the clusters largely depend on the problem domain and the specific dataset. Here are a few possible interpretations:

Outliers and Noise: The '-1' label is assigned to noise by the DBSCAN algorithm. If a large number of instances are labeled as '-1', it might mean that your eps and min_samples parameters need to be adjusted, or it could be that there's a large amount of variability in your data that the algorithm is interpreting as noise.

Cluster Sizes: The varying sizes of the clusters could indicate different levels of generality in your data. Larger clusters might represent more general or popular combinations of features (e.g., commonly seen genres, popular release years), while smaller clusters could represent more niche or unusual combinations.

Genre Patterns: If the clusters were primarily based on genre, you might find that the clusters represent common combinations of genres. For example, you might find that certain genres often appear together in the same game, and the clusters might represent these common combinations.

Year, Rating, and Voting Influence: Clustering on these variables along with genre might lead to clusters that represent different trends over time, different levels of game quality, or different levels of community engagement. For instance, there might be clusters of high-rated, highly-voted recent games in popular genres, and other clusters of low-rated older games.

Remember that these are just potential interpretations. The real meaning of the clusters can often only be understood by combining the clustering results with your domain knowledge about the data.

First options are to calculate the cluster and pull the cluster stats.

![Screenshot 2023-08-06 at 8 05 47 PM](https://github.com/jrfloza12/Project4/assets/122821004/c5b8554b-89a2-41e6-ac40-46a39c08e832)

![Screenshot 2023-08-06 at 8 06 04 PM](https://github.com/jrfloza12/Project4/assets/122821004/a3a9c300-636e-4bfb-ba15-ecd655c78628)

DBSCAN works best when you have dense regions of points separated by regions of low density. If DBSCAN is appropriate for the data, we might want to adjust our eps and min_samples parameters. A good way to start is by looking at a histogram or KDE plot of the pairwise distances in the dataset to help guide the choice of eps.

The process of looking at a histogram plot of pairwise distances involves calculating the distance between each pair of points in the dataset, and then plotting a histogram. We will use the pdist function from the scipy.spatial.distance module, which computes the pairwise distances, and matplotlib to plot the histogram. As suggested by GPT.

![Screenshot 2023-08-06 at 8 08 25 PM](https://github.com/jrfloza12/Project4/assets/122821004/f4afcc59-302a-4f93-97ea-841c34ef2786)

Cluster Stats show the following after more groubing.

![Screenshot 2023-08-06 at 8 11 19 PM](https://github.com/jrfloza12/Project4/assets/122821004/4d55869e-5462-4927-bca6-6123be3899a2)

We also created a dataframe to calculate the cluster size.
The cluster size gives us an idea of the Year Mean, Rating Mean and Votes.
Once we determined this we decided to join the two datasets.
This allowed us to compute the mean of each cluster:

![Screenshot 2023-08-06 at 8 18 09 PM](https://github.com/jrfloza12/Project4/assets/122821004/ae464cfa-ab80-481a-a738-7ac861aefa30)

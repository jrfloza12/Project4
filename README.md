# Project4
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
This particular data set contains the Name, Platform, Summary/Plot, Metacritic Rating and User Rating.
There is about___ rows of data. 

## ETL

We used Pandas for the dataframe to clean up the data prior to clustering and submitting a model for machine learning.
From all the data we wanted to identify three unique features "genres, esrb rating and platforms.
We created three csv files from this set, "[csv1](with_ratingvotes.csv) , [csv2](genreonly.csv) , [csv3](with_certificate.csv).

[ETL File](Cluster_ETL_2.ipynb)









## 

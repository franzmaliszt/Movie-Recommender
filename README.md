# movierecommender
# Movie Recommender Using Collaborative Filtering

## 1.	Purpose Of The Project
Aim is making prediction on how much a user would rate a movie, thus recommending a movie which they would like.

## 2.	Dataset
Movielens made available a data set of movie rating website for research. In this particular dataset with 100k entries, 3 main fields will be relevant in this project. These are user_id, movie_id, rating.

## 3.	Method
For this task, Collaborative Filtering is employed and implemented in two modes, namely user-based and item-based. We will divide it into some main steps.
###	Validation
As validation strategy, k-fold cross validation algorithm is used, in which the dataset is divided into k parts. The algorithm runs in k loops. In each iteration a part is chosen as test set and the rest k-1 part forms train set. In the end each entry would be tested once and trained k-1 time. 
If k decreases too much, test set will make up bigger percentage of all the data. Resulting in a train set that doesn’t represent the test set. Whereas if k increases and test set makes up the small percentage, we overestimate the skill of train set. Average error of each iteration makes up the final error evaluation.
###	Similarity Function
When making this prediction we need measured similarity between test item and train data. Because the prediction is made depending on other similar users or items. In general, item based model gives better accuracy than user based one. Because humans are inherently complex and brings in more bias.
###	Neighbor Selection
Since this is called collaborative, we need help of other similar user/items to come up with a prediction. When picking the most similar items, we have to find out what items would fit the role neighbor best. In this project, k-nn algorithm is used. 
Choosing small k leads to an unstable model which will be sensitive to irrelevant cases. But a greater k leads to poorer performance and it includes distant neighbors. Optimal k can be found after experiments because it varies with context.
###	User-Based CF
In User-based CF, we select every other user in trainset who rated the target movie. We calculate the similarity between test user with each selected user. Pearson correlation is used in user-based cf to calculate this. In which the ratings they gave to the common movies they both watched are used as metrics. Then k of the most similar users are used as final neighbors to help make prediction.
To make this more accurate, pearson correlation takes neighbor profile into consideration to reduce the bias soft and hard-raters cause. Each user’s average ratings are substracted from their actual ratings on test movie. Later adjusted again by their similarity weight. What yields meaning is; combining each user’s deviation for the rating they gave to this movie, and how much of relevance they have to test user by weighing them by their similarity.
###	Item-Based CF
In Item-based CF, similar approach is utilized. This time not user but items are used to create a model. To predict how much a user would give to movie, we find similar movies to the test movie. Every movie the test user has rated in trainset are selected and their similarity to the test movie is calculated.
Cosine similarity is utilized for this task. However, cosine similarity acts if movies yet not seen are voted 0. K nearest movies which the user already rated are used as neighbors. Their ratings are again adjusted by their similarities. Finally a prediction is made.
## 4.	Evaluation
The accuracy of predictions are evaluated using Mean Absolute Error.

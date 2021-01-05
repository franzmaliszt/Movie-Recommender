import statistics
import math

from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate


def load_movies(p):
    ml = []
    with open(p, 'r') as f:
        for line in f:
            column = line.strip().split('\t')
            user_id = int(column[0])
            item_id = int(column[1])
            rating = int(column[2])

            ml.append((user_id, item_id, rating))

    return ml


def user_similarity(u1, u2):
    mean_u1 = statistics.mean(u1.values())
    mean_u2 = statistics.mean(u2.values())

    commons = set(u1.keys()).intersection(set(u2.keys()))                       # Get the intersection of two users

    dividend = sum([(u1[c]-mean_u1) * (u2[c]-mean_u2) for c in commons])        # For each common movie, get the sum of multiplication of adjusted ratings

    dvr1 = math.sqrt(sum([(u1[c]-mean_u1)**2 for c in commons]))                # For each common movie, sqrt of sum of each movie's adjusted rating's square
    dvr2 = math.sqrt(sum([(u2[c]-mean_u2)**2 for c in commons]))

    divisor = dvr1 * dvr2                                                       # Multiply the squareroots

    try:
        return dividend/divisor
    except ZeroDivisionError:
        return 0


def user_based(train, test, knn):
    ratings, similarities = defaultdict(lambda: dict()), defaultdict(lambda: dict())
    truth, predictions = [], []

    for user_id, item_id, rating in train:
        ratings[user_id][item_id] = rating

    for user_id, item_id, rating in test:
        truth.append(rating)                                                                        # Real ratings given

        user_id_ratings = ratings[user_id]
        others = [k for k in ratings.keys() if k != user_id]                                        # List version of ratings without test user
        for o in others:
            if o not in similarities[user_id]:                                                      # Create similarities matrix check if it is already there   TestUser X TrainUsers
                similarities[user_id][o] = user_similarity(user_id_ratings, ratings[o])             # Calculate user_id's similaritiy to every other user o

        relative = []
        for i in similarities[user_id].items():                                                     # For each similar user, similarity
            if item_id in ratings[i[0]]:                                                            # See if similar user already voted on the test item
                relative.append(i)                                                                  # List of users rated the same movie.
        nearest = sorted(relative, key=lambda x: x[1], reverse=True)[:knn]                          # Sort the list and get the most similar knn users. (user_id,similarity) sorted by similarity

        p = user_prediction(user_id, item_id, ratings, nearest)                                     # Test user, Test item, trained data, users who rated the same test movie with ratings
        predictions.append(p)                                                                       # Add the predicted rating in a list.

    return mean_absolute_error(truth, predictions)                                                  # Get mae for all predictions in train set


def user_prediction(user, movie, ratings, neighbors):
    dividend = sum([n[1] * (ratings[n[0]][movie] - statistics.mean(ratings[n[0]].values())) for n in neighbors])        # the sum for each neighbor's multiplications of similarity with (rating for test movie - average neighbor ratings)
    divisor = sum([n[1] for n in neighbors])                                                                            # the sum of each neighbor's ratings

    try:
        return statistics.mean(ratings[user].values()) + (dividend / divisor)                                           # The average of test user's train ratings + dividend/divisor
    except ZeroDivisionError:
        return statistics.mean(ratings[user].values())


def item_similarity(i1, i2):
    dividend = sum([i1[j]*i2[j] for j in i1 if j in i2.keys()])

    dvr1 = math.sqrt(sum([(i1[m])**2 for m in i1]))                # For each common movie, sqrt of sum of each movie's adjusted rating's square
    dvr2 = math.sqrt(sum([(i2[m])**2 for m in i2]))

    divisor = dvr1 * dvr2                                                       # Multiply the squareroots

    try:
        return dividend/divisor
    except ZeroDivisionError:
        return 0


def item_based(train, test, knn):
    ratings, ratings_t = defaultdict(lambda: dict()), defaultdict(lambda: dict())
    truth, predictions = [], []

    for user_id, item_id, rating in train:
        ratings[item_id][user_id] = rating
        ratings_t[user_id][item_id] = rating

    for user_id, item_id, rating in test:
        truth.append(rating)                                                                        # Real ratings given
        similarities = defaultdict(lambda: dict())

        rated = ratings_t[user_id]                                                                  # Every movie the test user has rated
        if item_id in rated:                                                                        # Remove if test movie is already rated
            del rated[item_id]

        item_id_ratings = ratings[item_id]                                                          # Every rating the test item has
        for o in rated:
            if o not in similarities[item_id]:                                                      # Create similarities matrix and check if it is already there   TestItem X TrainItem
                similarities[item_id][o] = item_similarity(item_id_ratings, ratings[o])             # Calculate item_id's similaritiy to every other item o which test user already voted

        nearest = sorted(similarities[item_id].items(), key=lambda x: x[1], reverse=True)[:knn]     # Sort by the similarity and get knn of them

        commons = []
        for k in nearest:
            commons.append((k[0], rated[k[0]]))                                                     # Create a list with item_id and rating

        p = item_prediction(nearest, commons)                                                       # nearest= similar movies with similarity, commons= similar movies with ratings
        predictions.append(p)                                                                       # Add the predicted rating in a list.

    return mean_absolute_error(truth, predictions)                                                  # Get mae for all predictions in train set


def item_prediction(neighbors, rated):
    nei = [n[1] for n in neighbors]                                                                 # list of neighbors' ratings
    rat = [r[1] for r in rated]                                                                     # list of user's ratings

    dividend = sum([rat[m]*nei[m] for m in range(len(nei))])
    divisor = sum([n for n in nei])

    try:
        return dividend / divisor
    except ZeroDivisionError:
        return 2.5


def present(model, knn, results):
    rows = []

    for i in range(len(results)):
        rows.append([model, knn, i+1, results[i]])

    rows.append([model, knn, "Average", statistics.mean(results)])

    print(tabulate(rows, headers=["Model", "KNN", "Fold", "MAE"]))


if __name__ == '__main__':
    # path = "data/u.data"
    # mode = "user"
    # kf = KFold(n_splits=10)
    # k_nn = int(10)

    print("Data path: ")
    path = input("Please enter the path to the ml-100k data source: ")

    print("CF Algorithm to be used: ")
    mode = input("What mode of CF approach do you want? (user/item)")

    print("k-fold Cross Validation: ")
    kf = KFold(n_splits=int(input("Please set k_fold amount: ")))

    print("k Nearest neighbor approach: ")
    k_nn = int(input("Please set k_nn: [10, 20, 30, 40, 50, 60, 70, 80]"))

    movies = load_movies(path)
    maes = []
    for train_index, test_index in kf.split(movies):
        train_data = []
        for i in train_index:
            train_data.append(movies[i])

        test_data = []
        for i in test_index:
            test_data.append(movies[i])

        if mode == "user":
            mae = user_based(train_data, test_data, k_nn)
            maes.append(mae)
        else:
            mae = item_based(train_data, test_data, k_nn)
            maes.append(mae)

    present(results=maes, knn=k_nn, model=mode)

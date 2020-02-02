import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from models import *


def hit_ratio(x, model):
    X = x.movieId[x.val_query]
    y = x.movieId[x.val_query + 1]
    if y in model.predict(X, k=10):
        return 1
    else:
        return 0


def create_val_query(x):
    idx = random.randrange(len(x) - 1)
    return idx


def plot(dict_res, path='comparisson.png'):
    sns.set()
    plt.figure(figsize=(15, 12))
    plt.title('Compassion of algorithm by hit ratio')
    plt.bar(x=range(len(dict_res.values())), height=list(dict_res.values()), color=list('rgb'))
    plt.xticks(range(len(dict_res.keys())),list(dict_res.keys()))
    plt.show()
    plt.savefig(path)


def prepare_data(path_movies='../ml-20m/movies.csv',
                 path_ratings='../ml-20m/ratings.csv', min_rating=2,
                 min_watched=10):
    movies = pd.read_csv(path_movies)
    ratings = pd.read_csv(path_ratings)
    overall = ratings.merge(movies, right_on='movieId', left_on='movieId', how='inner')
    overall = overall[overall['rating'] >= min_rating]
    to_validate = overall.groupby('userId').agg({'movieId': list})
    to_validate = to_validate[to_validate['movieId'].apply(len) >= min_watched]
    to_validate['val_query'] = to_validate['movieId'].apply(lambda x: create_val_query(x))
    return to_validate, overall, movies


def ab_test(path_to_recs, path_movies='../ml-20m/movies.csv',
            path_ratings='../ml-20m/ratings.csv', min_rating=2,
            min_watched=10):
    recs = pd.read_json(path_to_recs)
    to_validate, overall, movies = prepare_data(path_movies, path_ratings, min_rating, min_watched)
    rel_model = RelatedModel(recs)
    rand_model = RandomModel(movies)
    most_pop_model = MostPopModel(overall)
    dict_res = {}
    for model, name in zip([rel_model, rand_model, most_pop_model],
                           ['related_model', 'random_model', 'most_popular_model']):
        dict_res.update({name: to_validate.apply(lambda x: hit_ratio(x, model), axis=1).mean()})
    plot(dict_res)

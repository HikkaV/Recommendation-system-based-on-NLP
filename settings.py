import sys
from contextlib import closing
from multiprocessing.dummy import Pool
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from urllib import request
import gzip
import io
import logging
import os

urls_imdb = {'url_imdb_genres': 'https://datasets.imdbws.com/title.basics.tsv.gz',
             'url_imdb_characters': 'https://datasets.imdbws.com/title.principals.tsv.gz',
             'url_imdb_directors': 'https://datasets.imdbws.com/title.crew.tsv.gz',
             'url_imdb_mapping': 'https://datasets.imdbws.com/name.basics.tsv.gz'}

weights = {'title': 10, 'genres': 6, 'directors': 2, 'writers': 1, 'cast': 4, 'characters': 3, 'titleType': 2}
fields = ['title', 'genres', 'directors', 'writers', 'cast', 'characters', 'titleType']
weights_space = [
    list(range(0, 11)) for i in range(0, len(weights.keys()))
]
# weights = {'title': 1, 'genres': 1, 'directors': 10, 'writers': 10, 'cast': 10, 'characters': 10, 'titleType': 1}
columns_to_vectorize = ['genres', 'characters', 'directors', 'writers', 'cast', 'titleType']
columns_for_embeddings = ['title']
path_to_alg_data = '../data/algorithm_data.csv'
path_to_overall_data = '../data/overall_data_initial_weights.csv'
path_to_model = '../GoogleNews-vectors-negative300.bin'
path_to_movies = '../ml-20m/movies.csv'
path_to_links = '../ml-20m/links.csv'
path_to_ratings = '../ml-20m/ratings.csv'
path_to_recs = '../data/overall_data_initial_weights_recs.json'
min_rating = 2
min_watched = 10
workers = 8
topn = 50
batch_size = 1000
ncalls = 10

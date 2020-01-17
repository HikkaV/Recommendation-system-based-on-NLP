import argparse
from settings import *
from related import Related
from predict import Predict
from preprocessing import Preprocess

parser = argparse.ArgumentParser()


def parse_args():
    subparsers = parser.add_subparsers()
    preprop = subparsers.add_parser('preprocess', help='Preprocess data for generating '
                                                       'related predictions')
    preprop.add_argument('-alg', dest='path_to_alg', help='Path to algorithm data', required=False,
                         default=path_to_alg_data, type=str)
    preprop.add_argument('-overall', dest='path_to_overall', help='Path to overall'
                                                                  ' interactive data',
                         required=False, default=path_to_overall_data, type=str)
    preprop.add_argument('-links', dest='path_to_links', help='Path to additional movielens data with links',
                         required=False, default=path_to_links, type=str)
    preprop.add_argument('-initial', dest='path_to_movies', help='Path to initial movielens data',
                         required=False, default=path_to_movies, type=str)
    preprop.set_defaults(func=preprocess)

    pred = subparsers.add_parser('predict', help='Predict topN related items giving tconst id')
    pred.add_argument('-overall', dest='path_to_overall', help='Path to overall'
                                                               ' interactive data',
                      required=False, default=path_to_overall_data, type=str)
    pred.add_argument('-id', dest='tconst', help='The id of the item the predictions '
                                                 'will be generated for',
                      required=True, type=str)
    pred.add_argument('-mode', dest='display', help='Display predictions if True', type=bool,
                      default=False, required=False)
    pred.set_defaults(func=predict)
    generate_parser = subparsers.add_parser('generate_related', help='Generate related '
                                                                     'from data prepared '
                                                                     'for algorithm')
    generate_parser.add_argument('-model', dest='model_path', help='Path to word2vec model '
                                                                   'used to retrieve predictions',
                                 required=False, default=path_to_model, type=str)
    generate_parser.add_argument('-alg', dest='path_to_alg', help='Path to algorithm data', required=False,
                                 default=path_to_alg_data, type=str)
    generate_parser.add_argument('-overall', dest='path_to_overall', help='Path to overall'
                                                                          ' interactive data',
                                 required=False, default=path_to_overall_data, type=str)
    generate_parser.set_defaults(func=generate_related)

    args = parser.parse_args()
    return args


def generate_related(args):
    rel = Related(model_path=args.model_path, path_to_alg=args.path_to_alg)
    rel.create_related(args.path_to_overall)


def preprocess(args):
    prep = Preprocess(path_to_links=args.path_to_links, path_to_movies=args.path_to_movies)
    prep.preprocess(path_show_data=args.path_to_overall, path_alg_data=args.path_to_alg)


def predict(args):
    pred = Predict(path_to_data=args.path_to_overall)
    pred.predict(args.tconst, args.display)

from skopt import forest_minimize
from related import Related
from metrics import hit_ratio, prepare_data
from models import RelatedModel


class Optimize:
    def __init__(self, space, model_path, path_to_alg, path_to_overall_data,
                 path_movies, path_ratings, min_ratings, min_watched, fields):
        self.space = space
        self.model_path = model_path
        self.path_to_alg = path_to_alg
        self.path_to_overall_data = path_to_overall_data
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.min_ratings = min_ratings
        self.min_watched = min_watched
        self.fields = fields

    def objective(self, space):
        weights = dict(list(zip(self.fields, space)))
        self.rel_cls.create_related(path_to_actual_data=self.path_to_overall_data, weights_specific=weights)
        data = self.rel_cls.algorithm_data
        related_model = RelatedModel(data)
        to_validate, _, __ = prepare_data(path_movies=self.path_movies, path_ratings=self.path_ratings,
                                          min_watched=self.min_watched, min_rating=self.min_ratings)
        return -to_validate.apply(lambda x: hit_ratio(x, related_model), axis=1).mean()

    def minimize(self, ncalls=10):
        self.rel_cls = Related(model_path=self.model_path, path_to_alg=self.path_to_alg)
        best_params = forest_minimize(self.objective, dimensions=self.space, n_calls=ncalls,
                                      verbose=1)['x']
        weights = dict(list(zip(self.fields, best_params)))
        self.rel_cls.create_related(path_to_actual_data=self.path_to_overall_data, weights_specific=weights)
        print('Generated best recommendations')

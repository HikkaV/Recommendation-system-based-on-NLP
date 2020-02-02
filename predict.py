from settings import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('../predict.log', 'a'))
print = logger.info


class Predict:
    def __init__(self, path_to_data):
        self.df = pd.read_json(path_to_data)

    def predict(self, tconst, display_mode=False):
        actual = self.df[self.df['tconst'] == tconst].copy()
        recs = list(map(tuple, actual.recs.values[0]))
        id_, rank = zip(*recs)
        pred = self.df[self.df.index.isin(list(id_))].copy()
        if display_mode:
            print("Actual item's meta data :")
            print('Title : {} , genres : {}, characters : {}, directors : {}, cast : {}'
                  'writers : {} , titleType : {}'.format(actual['title'].values[0],
                                                         actual['genres'].values[0],
                                                         actual['characters'].values[0],
                                                         actual['directors'].values[0],
                                                         actual['cast'].values[0],
                                                         actual['writers'].values[0],
                                                         actual['titleType'].values[0]))
            print('\n')
            print("Top {} predicted items :".format(topn))
            for i in pred.iterrows():
                print(i[1].loc['title'])
                print('Title : {} , genres : {}, characters : {}, directors : {}, cast : {}'
                      'writers : {} , titleType : {}'.format(i[1].loc['title'],
                                                             i[1].loc['genres'],
                                                             i[1].loc['characters'],
                                                             i[1].loc['directors'],
                                                             i[1].loc['cast'],
                                                             i[1].loc['writers'],
                                                             i[1].loc['titleType']))
                print('\n')
        return pred.tconst.values

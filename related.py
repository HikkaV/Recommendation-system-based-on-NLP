from settings import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('../related.log', 'a'))
print = logger.info


class Related:
    def __init__(self, model_path, path_to_alg):
        self.model = self.load_emb_from_disk(model_path)
        self.algorithm_data = pd.read_csv(path_to_alg)
        self.algorithm_data.fillna(value='nan', inplace=True)
        self.cv = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')

    def cv_matrices(self, columns):
        matrix = []
        for i in columns:
            print(i)
            matrix.append((weights[i], self.cv.fit_transform(self.algorithm_data[i])))
        return matrix

    def create_embs_dict(self, columns):
        dict_words = {}
        for i in columns:
            dict_words.update(self.get_rep_from_gensim(self.algorithm_data[i].values))
        print(len(dict_words))
        return dict_words

    def get_rep_from_gensim(self, words):
        dict_words = {}
        counter = []
        for w in tqdm(words):
            if not w in self.model.wv.vocab:
                w = w.lower()
            try:
                dict_words.update({w: self.model.word_vec(w)})

            except:
                counter.append(w)

        print('{} words were absent in embedding'.format(len(counter)))
        print('{} unique words were absent in embeddings'.format(len(set(counter))))
        print('Size of dict in RAM {} mb'.format(sys.getsizeof(dict_words) / 1000000))
        return dict_words

    def average(self, dict_words: dict, shape: int, words: list) -> np.array:
        avg = np.zeros(shape, dtype=np.float32)
        counter = 1
        for w in words:
            try:
                avg += dict_words[w]
                counter += 1
            except:
                pass
        counter-=1
        avg = avg / counter
        avg = np.nan_to_num(avg)
        return avg

    def multiple_avg(self, dict_words, representation):
        shape = len(list(dict_words.values())[0])
        matrix = lil_matrix((len(representation), shape), dtype=np.float32)
        for counter1, i in tqdm(enumerate(representation)):
            matrix[counter1, :] = self.average(dict_words, shape, i)

        return matrix

    def load_emb_from_disk(self, path):
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        return model

    def embs_matrices(self, dict_words, columns, matrix, weights):
        for i in columns:
            matrix.append((weights[i], self.multiple_avg(dict_words, self.algorithm_data[i].values)))
        return matrix

    def cswp(self, list_with_matrices, batch_size=5, top_n=50, num_workers=4):
        self.ranks = []
        m1 = list_with_matrices[0][1]
        kwargs_numpy_apply = {"top_n": top_n}
        start = 0
        if_counter = 0
        while start < m1.shape[0]:
            # define end
            if if_counter:
                break
            end = start + batch_size
            if end > m1.shape[0]:
                if_counter += 1
                end = m1.shape[0]
            with closing(Pool(num_workers)) as pool:
                tmp = pool.map(self._parallel_emb_cosine,
                               zip(list_with_matrices, [(start, end) for i in range(len(list_with_matrices))]))
                sim = np.sum(tmp, axis=0)
            sim = np.divide(sim, len(list_with_matrices))
            np.apply_along_axis(self.take_top_n, axis=1, arr=sim, **kwargs_numpy_apply)
            start = end
        return self.ranks

    def _parallel_emb_cosine(self, x):
        i, start_end_points = x
        rows = i[1][start_end_points[0]:start_end_points[1]]
        sim = cosine_similarity(rows, i[1])
        sim = np.multiply(sim, i[0])
        return sim

    def take_top_n(self, input_array, top_n):
        sim_scores = list(enumerate(input_array))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n]
        self.ranks.append([(i[0], self.sigmoid_log(i[1])) for i in sim_scores])

    def sigmoid_log(self, z, denominator=1):
        return 1. / (denominator + np.exp(np.negative(np.log(z))))

    def create_related(self, path_to_actual_data, weights_specific=None):
        matrix = self.cv_matrices(columns=columns_to_vectorize)
        dict_words = self.create_embs_dict(columns_for_embeddings)
        if not weights_specific:
            weights_specific = weights
        matrix = self.embs_matrices(dict_words, columns_for_embeddings, matrix, weights_specific)
        print('Made matrices')
        ranks = self.cswp(matrix, batch_size, topn, workers)
        print('Computed cosine similarity')
        self.algorithm_data = pd.read_csv(path_to_actual_data)
        print('Loaded data from {} to update with predictions'.format(path_to_actual_data))
        self.algorithm_data['recs'] = ranks
        path = path_to_actual_data.split('.csv')[0] + '_recs' + '.json'
        self.algorithm_data.to_json(path)
        print('Saved results to {0}'.format(path))

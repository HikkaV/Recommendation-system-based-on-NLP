from abc import abstractmethod
import numpy as np


class RecModel:
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def predict(self, X, k):
        '''
        Given input data X and number of topN results k -
        predict new data samples
        '''


class RandomModel(RecModel):
    def __init__(self, data):
        super().__init__(data)

    def predict(self, X, k=1):
        idx = np.random.randint(len(self.data) - 1, size=k)
        preds = self.data.loc[idx].movieId
        return preds


class MostPopModel(RecModel):
    def __init__(self, data, K=500):
        super().__init__(data)
        tmp = self.data.groupby('movieId').size().to_dict()
        self.sample_data = list(list(zip(*sorted(tmp.items(), key=lambda x: x[1], reverse=True)[:K]))[0])

    def predict(self, X, k=1):
        idx = np.random.choice(self.sample_data, size=k)
        preds = self.data.loc[idx].movieId
        return preds


class RelatedModel(RecModel):
    def __init__(self, data):
        super().__init__(data)

    def predict(self, X, k=1):
        preds = []
        if not isinstance(X, list):
            X = [X]
        for i in X:
            tmp = self.data[self.data['movieId'] == i]
            try:
                preds.extend(list(zip(*list(tmp["recs"].values)[0]))[0][:k])
            except:
                return []
        return preds



from settings import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('../preprocessing.log', 'a'))
print = logger.info


class Preprocess:
    def __init__(self, path_to_movies, path_to_links='ml-20m/links.csv'):
        self.make_dir()
        self.movies = pd.read_csv(path_to_movies)
        self.links = pd.read_csv(path_to_links)

    def make_dir(self):

        if not os.path.exists('../data/'):
            print("Dir data doesn't exist, thus making it")
            os.mkdir('../data/')
        else:
            print('Dir data exists')

    def save_intermidiate(self, path):
        self.movies.drop_duplicates(subset=['title', 'year'], inplace=True)
        self.movies.to_csv(path, index=False, mode='w')

    def make_years(self):
        self.movies['year'] = self.movies.title.apply(
            lambda x: ''.join(i for i in x.replace('–', '')[-6:] if i.isdigit()))

    def megrge_links(self):
        self.movies = self.movies.merge(self.links, left_on='movieId', right_on='movieId')

    def load_imdb(self, url):
        response = request.urlopen(url)
        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        return pd.read_csv(io.BytesIO(decompressed_file.read()), sep='\t')

    def padd(self, x) -> str:
        additional_str = 'tt'
        x = str(x)
        if len(x) < 7:
            for i in range(7 - len(x)):
                additional_str += '0'
        return additional_str + x

    def process_characters(self, x):
        x = list(set(x))
        if '\\N' in x:
            x.remove('\\N')
        x = [i.replace('[', '').replace(']', '').strip() for i in x]
        return x

    def get_all_nconst(self, merged):
        tmp = []
        for i in merged[['nconst', 'directors', 'writers']].values:
            tmp.append([x if isinstance(z, list) else z for z in i for x in z])

        tmp = list(set([x for z in tmp for x in z]))
        return tmp

    def mapping(self, x, imdb):
        final = []
        if isinstance(x, list):
            if '' in x:
                print(x)
                x.remove('')
            for i in x:
                if i in imdb.keys():
                    final.append(imdb[i])
        else:
            if not x == '':
                if x in imdb.keys():
                    final.append(imdb[x])
            else:
                pass
        return final

    def make_genres(self, x, y):
        if x == '(no genres listed)' or x == '\\N':
            x = 'nothing'
        genres_x = x.split('|') if '|' in x else [x]
        genres_y = y.split(',') if ',' in y else [y]
        final = list(set(genres_x).difference(genres_y)) + list(set(genres_x).intersection(genres_y)) + list(
            set(genres_y).difference(genres_x))
        if 'nothing' in final:
            final.remove('nothing')
        return final

    def enrich_genres(self):
        imdb = self.load_imdb(urls_imdb['url_imdb_genres'])
        self.movies['imdbId'] = self.movies['imdbId'].apply(lambda x: self.padd(x))
        self.movies = self.movies.merge(imdb, left_on='imdbId', right_on='tconst', how='left')
        self.movies.fillna(value='unknown', inplace=True)
        self.movies['genres'] = self.movies.apply(lambda x: self.make_genres(x.genres_x, x.genres_y), axis=1)
        self.movies = self.movies.drop(
            columns=['genres_y', 'genres_x', 'originalTitle', 'primaryTitle', 'tmdbId', 'runtimeMinutes',
                     'endYear', 'startYear', 'isAdult', 'imdbId'])

    def enrich_characters(self):
        imdb = self.load_imdb(urls_imdb['url_imdb_characters'])
        imdb = imdb[imdb['tconst'].isin(self.movies.tconst.values)]
        imdb = imdb.groupby('tconst').agg({'characters': list, 'nconst': list})
        imdb['characters'] = imdb['characters'].apply(lambda x: self.process_characters(x))
        self.movies = self.movies.merge(imdb, left_on='tconst', right_on='tconst', how='left')

    def enrich_directors_writers(self):
        imdb = self.load_imdb(urls_imdb['url_imdb_directors'])
        imdb = imdb[imdb['tconst'].isin(self.movies.tconst.values)]
        self.movies = self.movies.merge(imdb, left_on='tconst', right_on='tconst', how='left')
        self.movies.writers = self.movies.writers.apply(lambda x: str(x).split(',') if ',' in str(x) else x)
        self.movies.directors = self.movies.directors.apply(lambda x: str(x).split(',') if ',' in str(x) else x)
        self.movies.fillna(value='nan', inplace=True)

    def map_everything(self):
        imdb = self.load_imdb(urls_imdb['url_imdb_mapping'])
        imdb = imdb[['nconst', 'primaryName']]
        list_nconst = self.get_all_nconst(self.movies)
        imdb = imdb[imdb.nconst.isin(list_nconst)]
        imdb.index = imdb.nconst
        imdb.drop(columns='nconst', inplace=True)
        imdb = imdb.to_dict('dict')['primaryName']
        self.movies['cast'] = self.movies.nconst.apply(lambda x: self.mapping(x, imdb))
        self.movies['directors'] = self.movies.directors.apply(lambda x: self.mapping(x, imdb))
        self.movies['writers'] = self.movies.writers.apply(lambda x: self.mapping(x, imdb))
        self.movies.fillna(value='nan', inplace=True)

    def make_name_feature(self, x):
        final = []
        for i in x:
            final.append(i.replace(' ', ''))
        return final

    def clean_titles(self):
        self.movies['title'] = self.movies.title.str.replace(r'[(]\d{4}[)]?', '')
        self.movies['title'] = self.movies['title'].str.strip()
        self.movies.drop(columns=['nconst', 'tconst', 'movieId'], inplace=True)

    def stack_names(self):
        self.movies.writers = self.movies.writers.apply(lambda x: self.make_name_feature(x))
        self.movies.directors = self.movies.directors.apply(lambda x: self.make_name_feature(x))
        self.movies.cast = self.movies.cast.apply(lambda x: self.make_name_feature(x))

    def unpack_and_concat(self, list_with_data, concat=' '):
        if len(list_with_data) == 1:
            return list_with_data[0]
        return concat.join(list_with_data)

    def prepare_to_vectorize(self, df, columns):
        for i in columns:
            df[i] = df[i].apply(lambda x: self.unpack_and_concat(x))
        return df

    def preprocess(self, path_show_data, path_alg_data):
        self.make_years()
        print('Created years column')
        self.megrge_links()
        print('Merged with additional data containing tconst identifier')
        self.enrich_genres()
        print('Enriched genres')
        self.enrich_characters()
        print('Enriched characters')
        self.enrich_directors_writers()
        print('Enriched directors and writers')
        self.map_everything()
        print('Mapped all the nconsts to real names')
        self.save_intermidiate(path_show_data)
        print('Saved intermediate data for showing predictions later to {0}'.format(path_show_data))
        self.clean_titles()
        print('Cleaned titles in order to debias the algorithm')
        self.stack_names()
        print('Stacked names and surnames of cast as a new feature')
        self.movies = self.prepare_to_vectorize(self.movies, columns_to_vectorize[:-1])
        print('Unpacked columns')
        self.save_intermidiate(path_alg_data)
        print('Saved algorithmic data to {0}'.format(path_alg_data))

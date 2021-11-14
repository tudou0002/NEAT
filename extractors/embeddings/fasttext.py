from extractors.embeddings.embedding import Embedding
import fasttext
import pandas as pd

class FasttextEmbeddings(Embedding):
    """Use a pre-trained word embedding model 

    """
    def __init__(self, filename='extractors/src/fasttext_canadian_new.bin'):
        self.model = self.load_model(filename)
        self.dict = self.load_dict()

    def load_model(self, filename):
        return fasttext.load_model(filename)

    def load_dict(self):
        df_names = pd.read_csv('extractors/src/nameslist.csv')
        df_names.drop_duplicates(subset='Name', inplace=True)
        df_names.dropna()
        newwordlist = df_names['Name']
        return set([word.strip().lower() for word in newwordlist])

    def get_neighbors(self, word):
        """Return a list of strings include the top 10 words similar to the input word

        """
        neighbor_pairs = self.model.get_nearest_neighbors(word)
        return [word.encode('ascii','ignore').decode('ascii') for (similarity, word) in neighbor_pairs]

    def get_certainty(self, word):
        neighbors = self.get_neighbors(word)
        neighbors_in_dict = [word for word in neighbors if word in self.dict]
        return len(neighbors_in_dict) / len(neighbors)

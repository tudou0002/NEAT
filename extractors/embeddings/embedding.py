from abc import abstractmethod

class Embedding():
    # Interface for all the pretrained word embeddings
    def __init__(self):
        pass

    @abstractmethod
    def get_neighbors(self, word):
        """Return a list of words ordered by the similarity to the input word.

        """
        pass

    @abstractmethod
    def get_certainty(self, word):
        """Return a float that represents the number of neighbors in names dictionary / number of neighbors.

        """
        pass
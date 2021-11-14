from extractors.embeddings.fasttext import FasttextEmbeddings
import unittest 

class TestNameExtractor(unittest.TestCase):

    def setUp(self):
        self.embedding = FasttextEmbeddings()


    def test_get_neighbors(self):
        fuction_results = self.embedding.get_neighbors('alex')
        neighbors = ['alexbx', 'alexq', 'alexyc', 'alexia', 'alexxa', 'alexxia', 'alexax', 'alexy', 'alexi', 'alexa']
        
        self.assertEqual(fuction_results, neighbors)

    def test_get_certainty(self):
        fuction_result = self.embedding.get_certainty('alex')
        certainty = 0.4
        self.assertEqual(fuction_result, certainty)

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
from extractors.name_extractor import NameExtractor
from extractors.entity import Entity

class TestNameExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = NameExtractor()

    def test_basic(self):
        # two names are in the dictionary
        text = 'My name is vivian bella.'
        extractions = self.extractor.extract(text)

        entity_1 = Entity('vivian', 11)
        entity_2 = Entity('bella', 18)
        expected = [entity_1, entity_2]
        print(extractions)
        self.assertCountEqual(extractions, expected)

    def test_rule(self):
        # name is not in the dictionary but fits the rule
        text = 'My name is Hazel.'
        extractions = self.extractor.extract(text)

        expected = [Entity('Hazel', 11)]
        self.assertCountEqual(extractions, expected)


    def test_empty_str(self):
        text = ''
        extractions = self.extractor.extract(text)

        expected = []
        self.assertCountEqual(extractions, expected)


    def test_nan(self):
        text = np.nan
        extractions = self.extractor.extract(text)

        expected = []
        self.assertCountEqual(extractions, expected)

    
    def test_nameerror(self):
        self.assertRaises(NameError, NameExtractor, ['dictionary'], [''])



if __name__ == '__main__':
    unittest.main()
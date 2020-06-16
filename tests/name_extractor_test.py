import unittest
import numpy as np
from extractors.name_extractor import NameExtractor

class TestNameExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = NameExtractor()

    def test_basic(self):
        # names are in the dictionary
        text = 'My name is vivian bella.'
        extractions = self.extractor.extract(text)

        expected = ['vivian','bella']
        self.assertCountEqual(extractions, expected)

    def test_rule(self):
        # name is not in the dictionary but fits the rule
        text = 'My name is Hazel.'
        extractions = self.extractor.extract(text)

        expected = ['hazel']
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
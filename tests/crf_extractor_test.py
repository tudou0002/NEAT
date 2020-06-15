import unittest
from NameExtractor.extractors.crf_extractor import CRFExtractor

class TestCRFExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = CRFExtractor('test')

    def test_lower(self):
        text = 'My name is vivian.'
        extractions = self.extractor.extract(text)

        expected = ['Lisy']
        self.assertEqual(extractions, expected)


if __name__ == '__main__':
    unittest.main()

import spacy

class Extractor(object):
    """
    All extractors extend this abstract class.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")


    def extract(self, *input_value, **configs):
        """
        Args:
            input_value (): some extractors may want multiple arguments, for example, to
            concatenate them together
        Returns: list of extracted data as String. Returns an empty list if extractors fail
            to extract any data.
        """
        pass
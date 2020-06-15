class Extractor(object):
    """
    All extractors extend this abstract class.
    """

    def __init__(self, name: str = None):
        self._name = name


    @property
    def name(self) -> str:
        """
        The name of an extractor shown to users.
        Different instances ought to have different names, e.g., a glossary extractor for city_name could have
        name "city name extractor".
        Returns: string, the name of an extractor.
        """
        return self._name


    def extract(self, *input_value, **configs):
        """
        Args:
            input_value (): some extractors may want multiple arguments, for example, to
            concatenate them together
            configs (): any configs/options of extractors
        Returns: list of extracted data, which can be any Python object
        """
        pass
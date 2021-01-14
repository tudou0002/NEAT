from extractors.dict_extractor import DictionaryExtractor
from extractors.rule_extractor import RuleExtractor
from extractors.crf_extractor import CRFExtractor
from extractors.extractor import Extractor
import re, unicodedata
from extractors.filter import *

class NameExtractor(Extractor):
    def __init__(self, primary=['dict'],backoff=['rule'],w1=0.5,w2=0.5, threshold=0.3):
        """
        Initialize the extractor, storing the extractors types and backoff extractor types.
        Args:
            extractors (list): extractor types that will always apply extraction on the text
            backoff (list): extractor types that will only apply extraction on the text if all
                of the primary extractors failed to extract some names.
        Returns:
        """
        self.primary = self.initialize_extractors(primary)
        self.backoff = self.initialize_extractors(backoff)
        self.fillMaskFilter = FillMaskFilter()
        self.w1 = w1
        self.w2 = w2
        self.threshold = threshold

    
    def initialize_extractors(self, extractor_type:list):
        """
        Creates the extractors based on the given type
        Args:
            extractor_type (list): 'dict', 'rule',
        Returns:
            List(extractor): returns a list of extractor object or 
                             empty list if the extractor_type is an empty list
        """
        result_extractors = []
        for extractor in extractor_type:
            if extractor == 'dict':
                result_extractors.append(DictionaryExtractor())
            elif extractor == 'rule':
                result_extractors.append(RuleExtractor())
            elif extractor == 'crf':
                result_extractors.append(CRFExtractor())
            else:
                raise NameError("Invalid extractor type! The extractor type input must be 'dict', 'rule' or 'crf'")

        return result_extractors

    def extract(self, text, preprocess=True):
        """
            Extracts information from a text using the given extractor types.
        Args:
            text (str): the text to extract from.
            preprocess(bool): True if needed preprocessing
        Returns:
            List(str): the list of entities or the empty list if there are no matches.
        """
        results = []
        if preprocess:
            text = self.preprocess(text)
        for ext in self.primary:
            results.extend(ext.extract(text))
        # if the primary extractors fail to extract names, use the back off extractors
        if results==[]:
            for ext in self.backoff:
                results.extend(ext.extract(text))
        results_text = [result.text for result in results]
        filtered_results = self.fillMaskFilter.disambiguate_layer(text, results_text)

        # add the disambiguated ratio
        for result, filtered in zip(results, filtered_results):
            result.context_confidence = filtered['ratio']
            result.confidence = self.w1*result.confidence + self.w2*result.context_confidence
            if result.confidence < self.threshold:
                results - result

        return list(set(results))

    
    def preprocess(self, text):
        """
            Preprocesses the text: expanding contractions, removing emojis and punctuation marks
        Args:
            text (str): the text to be preprocessed
        Returns:
            str: the text after being preprocessed
        """
        CONTRACTION_MAP = {
            'names': 'name is',
            'its': 'it is',
            "I'm": "I am",
            "i'm": "I am",
            "name's": "name is",
            "it's": "it is",  
            "I've":"I have",
            "i've": "I have",
            "we've":'We have'
        }

        EMOJI_PATTERN = re.compile(
            "["
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+"
        )

        def replace_contraction(text):
            contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())), 
                                            flags=re.IGNORECASE)
            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = CONTRACTION_MAP.get(match)\
                                        if CONTRACTION_MAP.get(match)\
                                        else CONTRACTION_MAP.get(match.lower())                       
                expanded_contraction = first_char+expanded_contraction[1:]
                return expanded_contraction
                
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
            return expanded_text

    
        # return empty string if text is NaN
        if type(text)==float:
            return ''
        # remove emoji
        text = re.sub(EMOJI_PATTERN, r' ', text)
        text = re.sub(r'·', ' ', text)
        # convert non-ASCII characters to utf-8
        text = unicodedata.normalize('NFKD',text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = re.sub(r'<.*?>', ' ', text)
        text = replace_contraction(text)
        text = re.sub(r'[\'·\"”#$%&’()*+/:;<=>@[\]^_`{|}~-]+',' ',text)
        text = re.sub(r'[!,.?]{2,}\s?',' ',text)
        text = re.sub(r'[\s]+',' ',text)
        return text


        
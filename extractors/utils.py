import re, unicodedata
import truecase

def preprocess(text):
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
        text = truecase.get_true_case(text)
        return text
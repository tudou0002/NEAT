import spacy
import os
import pandas as pd
from extractors.extractor import Extractor
from spacy.matcher import PhraseMatcher
from extractors.entity import Entity
from extractors.embeddings.fasttext import FasttextEmbeddings

class DictionaryExtractor(Extractor):
    def __init__(self,
                 dict_file='extractors/src/nameslist.csv'):
        Extractor.__init__(self)
        self.terms = self.load_word_dict(dict_file)
        self.matcher = self.create_matcher()
        self.embedding = FasttextEmbeddings()


    def load_word_dict(self,dict_file): 
        df_names = pd.read_csv(dict_file)
        df_names.drop_duplicates(subset='Name', inplace=True)
        df_names.dropna()
        newwordlist = df_names['Name']
        return [word.strip().lower() for word in newwordlist]
    

    def create_matcher(self):
        matcher = PhraseMatcher(self.nlp.vocab,attr="LOWER")
        patterns = [self.nlp.make_doc(text) for text in self.terms]
        matcher.add('namelist', None, *patterns)
        return matcher
        
    def extract(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        result = []
        for match_id, start, end in matches:
            span = doc[start:end]
            ent = Entity(span.text,span.start)
            ent.score = self.embedding.get_certainty(ent.text)
            result.append(ent)
        return result
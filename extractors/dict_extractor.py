import spacy
import os, json
import pandas as pd
from extractors.extractor import Extractor
from spacy.matcher import PhraseMatcher
from extractors.entity import Entity
from extractors.embeddings.fasttext import FasttextEmbeddings

class DictionaryExtractor(Extractor):
    def __init__(self,**kwargs):
        model = kwargs.pop('model', 'en_core_web_sm')
        dict_file = kwargs.pop('dict_file', 'extractors/src/nameslist.csv')
        Extractor.__init__(self, model)
        try:
            self.terms = kwargs.pop('dictionary')
        except:
            self.terms = self.load_word_dict(dict_file)
        try:
            weights_file = kwargs.pop('weights_dict')
            self.weights = self.load_weight_dict(weights_file)
        except:
            self.weights = {n:0.5 for n in self.terms}
        self.matcher = self.create_matcher()
        self.type = 'dict'

    def load_weight_dict(self, filename):
        with open(filename) as json_file:
            weights = json.load(json_file)
        return weights

    def load_word_dict(self,dict_file): 
        df_names = pd.read_csv(dict_file)
        df_names.drop_duplicates(subset='Name', inplace=True)
        df_names.dropna()
        newwordlist = df_names['Name']
        return list(set([word.strip().lower() for word in newwordlist]))
    

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
            ent = Entity(span.text,span.start, self.type)
            ent.base_conf = self.weights[ent.text.lower()]
            ent.confidence = ent.base_conf
            ent.type = 'dict'
            result.append(ent)
        return result
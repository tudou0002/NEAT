import spacy
from spacy.matcher import Matcher
from extractors.extractor import Extractor
from extractors.entity import Entity
from extractors.embeddings.fasttext import FasttextEmbeddings

class RuleExtractor(Extractor):
    def __init__(self, model='en_core_web_sm'):
        Extractor.__init__(self, model)
        self.patterns = self.define_patterns()
        self.matcher = self.create_matcher(self.patterns)
        # self.embedding = FasttextEmbeddings()
        self.type = 'rule'
        
    def create_matcher(self,  patterns):
        matcher = Matcher(self.nlp.vocab)
        matcher.add('thirdMatch',  self.patterns[0])
        matcher.add('secondMatch',  self.patterns[1])
        return matcher
    
    def define_patterns(self):
        # create patterns
        pattern0 = [{"LOWER": "call"}, {"LOWER": "me"},{"TAG": "NNP"}]
        pattern1 = [{"LOWER": "name"}, {"LOWER": "is"},{"TAG": "NNP"}]
        pattern2 = [{"LOWER": "i"}, {"LOWER":"am"},{"TAG": "NNP"}]
        pattern3 = [{"LOWER": "it"}, {"LOWER":"is"},{"TAG": "NNP"}]
        pattern4 = [{"LOWER": "ask"}, {"LOWER":"for"},{"TAG": "NNP"}]
        pattern5 = [{"LOWER":"Ms"},{"TAG": "NNP"}]
        pattern6 = [{"LOWER":"ms."},{"TAG": "NNP"}]
        pattern7 = [{"LOWER":"aka"},{"TAG": "NNP"}]
        pattern8 = [{"LOWER":"miss"},{"TAG": "NNP"}]
        pattern9 = [{"LOWER":"Miss."},{"TAG": "NNP"}]
        pattern10 = [{"LOWER":"Ts"},{"TAG": "NNP"}]
        pattern11 = [{"LOWER":"Mrs"},{"TAG": "NNP"}]
        pattern12 = [{"LOWER":"mrs."},{"TAG": "NNP"}]
        pattern13 = [{"LOWER":"Mz"},{"TAG": "NNP"}]
        pattern14 = [{"LOWER":"mz."},{"TAG": "NNP"}]
        pattern15 = [{"LOWER":"named"},{"TAG": "NNP"}]
    
        return ([pattern0,pattern1, pattern2, pattern3, pattern4],
                [pattern5, pattern6, pattern7,pattern8,pattern9,pattern10,
                pattern11,pattern12,pattern13,pattern14,pattern15])
    
    def extract(self, text):
        if type(text)==float:
            return []
        doc = self.nlp(text)
        matches = self.matcher(doc)
        result = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]
            if string_id=='thirdMatch':
                name_start = start + 2
            else:
                name_start = start + 1
            span = doc[name_start:end] 
            ent = Entity(span.text,span.start, self.type)
            # ent.score = self.embedding.get_certainty(ent.text)
            ent.confidence = 0.4
            result.append(ent)
        return result
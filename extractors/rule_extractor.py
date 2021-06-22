import spacy
from spacy.matcher import Matcher
from extractors.extractor import Extractor
from extractors.entity import Entity
from extractors.embeddings.fasttext import FasttextEmbeddings

class RuleExtractor(Extractor):
    def __init__(self, **kwargs):
        model = kwargs.pop('model', 'en_core_web_sm')
        Extractor.__init__(self, model)
        self.patterns,self.weights = self.define_patterns()
        self.matcher = self.create_matcher(self.patterns)
        self.type = 'rule'
        
    def create_matcher(self,  patterns):
        matcher = Matcher(self.nlp.vocab)
        for k,v in self.patterns.items():
            matcher.add(k,[v])
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
    
        patterns={'pattern0':pattern0, 'pattern1':pattern1,'pattern2':pattern2,
               'pattern3':pattern3,'pattern4':pattern4,'pattern5':pattern5,
               'pattern6':pattern6,'pattern7':pattern7,'pattern8':pattern8,
               'pattern9':pattern9,'pattern10':pattern10,'pattern11':pattern11,
               'pattern12':pattern12,'pattern13':pattern13,'pattern14':pattern14,
                'pattern15':pattern15}
        weights = {'pattern0':(2, 0.2), 'pattern1':(2, 0.77),'pattern2':(2, 0.62),
               'pattern3':(2, 0.47),'pattern4':(2, 0.11),'pattern5':(1, 0.47),
               'pattern6':(1, 0.47),'pattern7':(1, 0.4),'pattern8':(1, 0.54),
               'pattern9':(1, 0.47),'pattern10':(1, 0.47),'pattern11':(1, 0.47),
               'pattern12':(1, 0.66),'pattern13':(1, 0.47),'pattern14':(1, 0.47),
                'pattern15':(1, 0.47)}
        return (patterns,weights)
        
    def extract(self, text):
        if type(text)==float:
            return []
        doc = self.nlp(text)
        matches = self.matcher(doc)
        result = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]
            name_start=start+self.weights[string_id][0]
            span = doc[name_start:end] 
            ent = Entity(span.text,span.start, self.type)
            ent.confidence = self.weights[string_id][1]
            result.append(ent)
        return result
import spacy
from spacy.matcher import Matcher
from extractors.extractor import Extractor
from extractors.entity import Entity

class RuleExtractor(Extractor):
    def __init__(self, **kwargs):
        model = kwargs.pop('model', 'en_core_web_sm')
        Extractor.__init__(self, model)
        self.patterns, self.weights = self.define_patterns()
        self.matcher = self.create_matcher()
        self.type = 'rule'
        
    def create_matcher(self):
        matcher = Matcher(self.nlp.vocab)
        for k,v in self.patterns.items():
            matcher.add(k,[v])
        return matcher
    
    def define_patterns(self):
        # English patterns
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

        # French patterns
        pattern16 = [{"LOWER": "appelez"}, {"LOWER": "moi"},{"TAG": "NNP"}]  # call me NNP
        pattern17 = [{"LOWER": "appelle"}, {"LOWER": "moi"},{"TAG": "NNP"}] # call me NNP
        pattern18 = [{"LOWER": "nom"}, {"LOWER": "est"},{"TAG": "NNP"}] # name is NNP
        pattern19 = [{"LOWER": "m"}, {"LOWER": "appelle"},{"TAG": "NNP"}]   # name is NNP
        pattern20 = [{"LOWER": "c"}, {"LOWER": "est"},{"TAG": "NNP"}]   # it is NNP
        pattern21 = [{"LOWER":"demander"},{"TAG": "NNP"}]   # ask for NNP
        pattern22 = [{"LOWER":"Mme"},{"TAG": "NNP"}]    # Ms NNP
        pattern23 = [{"LOWER":"Madame"},{"TAG": "NNP"}] # Ms NNP
        pattern24 = [{"LOWER":"Mademoiselle"},{"TAG": "NNP"}]   # Miss NNP
        pattern25 = [{"LOWER":"alias"},{"TAG": "NNP"}]  # aka NNP
        pattern26 = [{"LOWER":"surnom"},{"TAG": "NNP"}] # aka NNP
    
        patterns={'pattern0':pattern0, 'pattern1':pattern1,'pattern2':pattern2,
               'pattern3':pattern3,'pattern4':pattern4,'pattern5':pattern5,
               'pattern6':pattern6,'pattern7':pattern7,'pattern8':pattern8,
               'pattern9':pattern9,'pattern10':pattern10,'pattern11':pattern11,
               'pattern12':pattern12,'pattern13':pattern13,'pattern14':pattern14,
                'pattern15':pattern15,'pattern16':pattern16,'pattern17':pattern17,
                'pattern18':pattern18,'pattern19':pattern19,'pattern20':pattern20,
                'pattern21':pattern21,'pattern22':pattern22,'pattern23':pattern23,
                'pattern24':pattern24,'pattern25':pattern25,'pattern15':pattern26,}
        weights = {'pattern0':(2, 0.5), 'pattern1':(2, 0.67),'pattern2':(2, 0.44),
               'pattern3':(2, 0.35),'pattern4':(2, 0.72),'pattern5':(1, 0.5),
               'pattern6':(1, 0.5),'pattern7':(1, 0.5),'pattern8':(1, 0.67),
               'pattern9':(1, 0.5),'pattern10':(1, 0.5),'pattern11':(1, 0.5),
               'pattern12':(1, 0.5),'pattern13':(1, 0.5),'pattern14':(1, 0.5),
                'pattern15':(1, 0.75),'pattern16':(2, 0.5),'pattern17':(2, 0.5),
                'pattern18':(2, 0.67),'pattern19':(2, 0.67),'pattern20':(2, 0.35),
                'pattern21':(1, 0.72),'pattern22':(1, 0.5),'pattern23':(1, 0.5),
                'pattern24':(1, 0.5),'pattern25':(1, 0.5),'pattern15':(1, 0.5),}
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
            ent.base_conf = self.weights[string_id][1]
            ent.confidence = ent.base_conf
            ent.type = 'rule'
            result.append(ent)
        return result
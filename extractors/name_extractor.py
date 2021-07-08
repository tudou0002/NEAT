from extractors.dict_extractor import DictionaryExtractor
from extractors.rule_extractor import RuleExtractor
from extractors.crf_extractor import CRFExtractor
from extractors.extractor import Extractor
from extractors.filter import *
from extractors.utils import *

class NameExtractor(Extractor):
    def __init__(self, primary=['dict'],backoff=['rule'], threshold=0.3, **kwargs):
        """
        Initialize the extractor, storing the extractors types and backoff extractor types.
        Args:
            extractors (list): extractor types that will always apply extraction on the text
            backoff (list): extractor types that will only apply extraction on the text if all
                of the primary extractors failed to extract some names.
        Returns:
        """
        self.primary = self.initialize_extractors(primary, **kwargs)
        self.backoff = self.initialize_extractors(backoff, **kwargs)
        self.fillMaskFilter = FillMaskFilter()
        self.threshold = threshold

    
    def initialize_extractors(self, extractor_type:list, **kwargs):
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
                result_extractors.append(DictionaryExtractor(**kwargs))
            elif extractor == 'rule':
                result_extractors.append(RuleExtractor(**kwargs))
            elif extractor == 'crf':
                result_extractors.append(CRFExtractor(**kwargs))
            else:
                raise NameError("Invalid extractor type! The extractor type input must be 'dict', 'rule' or 'crf'")

        return result_extractors

    def extract(self, text, preprocess_text=True):
        """
            Extracts information from a text using the given extractor types.
        Args:
            text (str): the text to extract from. Usually a piece of ad.
            preprocess(bool): True if needed preprocessing
        Returns:
            List(str): the list of entities or the empty list if there are no matches.
        """
        results = []
        if preprocess_text:
            text = preprocess(text)
        for ext in self.primary:
            results.extend(ext.extract(text))
        # if the primary extractors fail to extract names, use the back off extractors
        if results==[]:
            for ext in self.backoff:
                results.extend(ext.extract(text))

        # pass to the disambiguation layer        
        results_text = [result.text for result in results]
        # print('text:', text)
        text = re.sub(r'[\.,]+',' ',text)
        filtered_results = self.fillMaskFilter.disambiguate_layer(text, results_text)
        # print('text:', text)
        # print(results_text)
        # print(results)
        # print(filtered_results)

        # add the disambiguated ratio
        conf_dict = {} # key: entity   value: [confidence, fill_mask_conf]
        for result, filtered in zip(results, filtered_results):
            if result not in conf_dict:
                conf_dict[result] = [result.confidence, filtered['ratio'], [filtered['context']]]
            else:
                conf_dict[result][0]  *= result.confidence
                conf_dict[result][1]  *= filtered['ratio']
                conf_dict[result][2].append(filtered['context'])

        # print(conf_dict)
        entity_list = []
        for ent, conf_list in conf_dict.items():
            ent.confidence = conf_list[0]
            ent.fill_mask_conf = conf_list[1]
            ent.context = conf_list[2]
            entity_list.append(ent)
        return entity_list

    
    


        
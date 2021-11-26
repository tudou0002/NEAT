from extractors.dict_extractor import DictionaryExtractor
from extractors.rule_extractor import RuleExtractor
from extractors.extractor import Extractor
from extractors.filter import *
from extractors.utils import *

class NameExtractor(Extractor):
    def __init__(self, primary=['dict', 'rule'],backoff=[], threshold=0.12, **kwargs):
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
        self.dict_extractor = DictionaryExtractor(**kwargs)
        self.rule_extractor = RuleExtractor(**kwargs)
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
            # elif extractor == 'crf':
            #     result_extractors.append(CRFExtractor(**kwargs))
            else:
                raise NameError("Invalid extractor type! The extractor type input must be 'dict', 'rule' or 'crf'")

        return result_extractors

    def find_ent(self, ent, ent_list):
        for e in ent_list:
            if ent==e:
                return e
        return None

    def compute_combined(self, dict_res, rule_res):
        intersection = dict_res & rule_res
        [print(ent) for ent in intersection]
        unilateral = (dict_res - rule_res) | (rule_res - dict_res)
        [print(ent) for ent in unilateral]

        for res in intersection:
            res.base_conf = self.find_ent(res, dict_res).base_conf*0.5 + self.find_ent(res, rule_res).base_conf*0.5 
        for res in unilateral:
            res.base_conf = self.find_ent(res, unilateral).base_conf*0.5
                
        total_res = list(intersection | unilateral)
        
        return total_res

    

    def extract(self, text, preprocess_text=True):
        """
            Extracts information from a text using the given extractor types.
        Args:
            text (str): the text to extract from. Usually a piece of ad.
            preprocess(bool): True if needed preprocessing
        Returns:
            List(str): the list of entities or the empty list if there are no matches.
        """
        if preprocess_text:
            text = preprocess(text)
        dict_res = set(self.dict_extractor.extract(text))
        rule_res = set(self.rule_extractor.extract(text))
        # [print(ent) for ent in dict_res]
        # [print(ent) for ent in rule_res]
        # total_res = dict_res.extend(rule_res)
        # [print(ent) for ent in total_res]
        results = self.compute_combined(dict_res, rule_res)
        
        # pass to the disambiguation layer        
        results_text = [result.text for result in results]
        # print('text:', text)
        text = re.sub(r'[\.,]+',' ',text)
        filtered_results = self.fillMaskFilter.disambiguate_layer(text, results_text)
      

        # add the disambiguated ratio
        conf_dict = {} # key: entity   value: [confidence, fill_mask_conf, context]
        for result, filtered in zip(results, filtered_results):
            if result not in conf_dict:
                conf_dict[result] = [result.base_conf, filtered['ratio'], [filtered['context']]]
            else:
                conf_dict[result][0]  *= result.base_conf
                conf_dict[result][1]  *= filtered['ratio']
                conf_dict[result][2].append(filtered['context'])

        entity_list = []
        for ent, conf_list in conf_dict.items():
            ent.base_conf = conf_list[0]
            ent.fill_mask_conf = conf_list[1]
            ent.context = conf_list[2]
            ent.confidence = ent.base_conf*0.5+ent.fill_mask_conf*0.5
            entity_list.append(ent)

        # print([(ent.text, ent.base_conf, ent.fill_mask_conf) for ent in entity_list])
        # return [ent.text for ent in entity_list if ent.base_conf*0.5+ent.fill_mask_conf*0.5>=self.threshold]
        return entity_list

    
    


        
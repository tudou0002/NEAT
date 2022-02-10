from extractors.dict_extractor import DictionaryExtractor
from extractors.rule_extractor import RuleExtractor
from extractors.extractor import Extractor
from extractors.filter import *
from extractors.utils import *

class NameExtractor(Extractor):
    def __init__(self, threshold=0.12, **kwargs):
        """
        Initialize the dictionary and rule extractors
        Args:
            threshold: a float that controls the confidence score used in filtering out the output.
        Returns:
        """
        self.dict_extractor = DictionaryExtractor(**kwargs)
        self.rule_extractor = RuleExtractor(**kwargs)
        self.fillMaskFilter = FillMaskFilter()
        self.threshold = threshold

    def find_ent(self, target_word, ent_list):
        """
        Return the entity if it is the same as the target entity. Inputs should guarantee there will be a match.s
        Args:
            target_word: A string value that holds the word you want to search.
            ent_list: A set of Entities that you want to search from.
        Returns:
        """
        for e in ent_list:
            if target_word==e:
                return e
        return None

    def compute_combined(self, dict_res, rule_res):
        """
        Compute the confidence score for each predicted word from the base extractors.
        Args:
            dict_res: A set of Entities extracted from the dictionary extractor.
            rule_res: A set of Entities extracted from the rule extractor.
        Returns:
            A list that contians all unique Entities with the combined confidence from the base extractors.
        """
        intersection = dict_res & rule_res
        unilateral = (dict_res - rule_res) | (rule_res - dict_res)

        for res in intersection:
            res.base_conf = self.find_ent(res, dict_res).base_conf*0.5 + self.find_ent(res, rule_res).base_conf*0.5 
        for res in unilateral:
            res.base_conf = self.find_ent(res, unilateral).base_conf*0.5
                
        total_res = list(intersection | unilateral)
        
        return total_res

    def extract(self, text, preprocess_text=True):
        """
            Extracts information from a text using NEAT.
        Args:
            text (str): the text to extract from. Usually a piece of ad description or its title.
            preprocess(bool): set to True if the input text needs preprocessing before the extraction. Default is True.
        Returns:
            List(Entity): a list of entities or the empty list if there are no extracted names.
        """
        if preprocess_text:
            text = preprocess(text)
        dict_res = set(self.dict_extractor.extract(text))
        rule_res = set(self.rule_extractor.extract(text))
        results = self.compute_combined(dict_res, rule_res)
        
        # pass to the disambiguation layer        
        results_text = [result.text for result in results]
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
        # compute and record the confidence score in the "confidence" field
        for ent, conf_list in conf_dict.items():
            ent.base_conf = conf_list[0]
            ent.fill_mask_conf = conf_list[1]
            ent.context = conf_list[2]
            ent.confidence = ent.base_conf*0.5+ent.fill_mask_conf*0.5
            if ent.confidence >= self.threshold:
                entity_list.append(ent)

        return entity_list

    
    


        
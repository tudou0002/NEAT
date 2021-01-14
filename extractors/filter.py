# from tokenizers.implementations import ByteLevelBPETokenizer
# from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import pipeline
import pandas as pd

from spacy.lang.en import English

class FillMaskFilter:
    def __init__(self):
        # self.tokenizer = ByteLevelBPETokenizer(
        #     "ht_bert/vocab.json",
        #     "ht_bert/merges.txt",
        #     # model_max_length = 120, 
        # )
        # self.tokenizer_tokenizer.post_processor = BertProcessing(
        #     ("</s>", tokenizer.token_to_id("</s>")),
        #     ("<s>", tokenizer.token_to_id("<s>")),
        # )
        # self.tokenizer.enable_truncation(max_length=512)

        tokenizer = RobertaTokenizerFast.from_pretrained("ht_bert", max_len=512)
        config = RobertaConfig(
            vocab_size=52_000,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        model = RobertaForMaskedLM(config=config).from_pretrained("ht_bert")

        self.fill_mask = pipeline(
            "fill-mask",
            model="ht_bert",
            tokenizer="ht_bert",
            top_k=20,
        )
        
        df_names = pd.read_csv('extractors/src/nameslist.csv')
        df_names.drop_duplicates(subset='Name', inplace=True)
        df_names.dropna()
        newwordlist = df_names['Name']
        self.name_set = set([word.strip().lower() for word in newwordlist])

    # compute the ratio in the result
    def compute_ratio(self, fill_mask_sim, ne_result):
        in_dict_counter = 0
        total = len(fill_mask_sim)
        for r in fill_mask_sim:
            if r['token_str'].strip('Ġ').lower() in self.name_set:
                in_dict_counter += 1
            if r['token_str'].strip('Ġ').lower() == ne_result:
                total -= 1
                in_dict_counter -= 1

        return in_dict_counter / total

    def disambiguate_layer(self, context, words, window_size=5):
        '''
        context: preprocessed context 
        words: results from the extractor

        return: a list of dictionary with keys like sent, word, ratio
        '''
        # sanity check
        if not words:
            return []

        results = []
        for word in words:
            info_dict = {}
            info_dict['word'] = word.lower()

            # select the context window for the word
            context_list = context.lower().split()
            try:
                word_idx = context_list.index(word.lower())
            except:
                # print(context_list, word)
                return {}
            window = ' '.join(context_list[max(0,word_idx-window_size):min(len(context_list), word_idx+window_size)])
            
            window = window.replace(word.lower(), '<mask>',1)
            info_dict['context'] = window

            fill_mask_sim = self.fill_mask(window)
            ratio = self.compute_ratio(fill_mask_sim, word.lower())
            info_dict['ratio'] = ratio
            results.append(info_dict)

        return results
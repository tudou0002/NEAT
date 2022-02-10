import pandas as pd
import argparse
from spacy.lang.en import English
import math
import nltk
from nltk.corpus import stopwords 
import json
from gensim.models import Word2Vec 
from spacy.lang.en import English
import spacy
from spacy.tokenizer import Tokenizer
from collections import Counter
from extractors.utils import *
from extractors.name_extractor import NameExtractor
from extractors.dict_extractor import DictionaryExtractor
from extractors.rule_extractor import RuleExtractor


class W2V():
    def __init__(self, texts, seeds, namedict):
        # prepare the corpus, the trainset should be a list of token list
        nlp = English()
        self.tokenizer = nlp.tokenizer
        self.corpus = []
        cleaned_texts = [preprocess(t).lower() for t in texts]
        for doc in self.tokenizer.pipe(cleaned_texts, batch_size=50):
            self.corpus.append([token.text for token in doc])
        # train word2vec model
        self.model = self.train_model()
        self.seeds = seeds
        self.namedict = namedict
        
    def train_model(self):
        model = Word2Vec(min_count=1, window=2)
        model.build_vocab(self.corpus)
        model.train(self.corpus, total_examples=model.corpus_count, epochs=50, report_delay=1)

        return model
    
    def expand(self, topn=10, thres=0, verbose=0):
        candidate_pool = []
        for s in self.seeds:
            candidate_pool.extend([w for (w,p) in self.model.wv.most_similar(s,topn=topn)])
        # remove the words that already in the name dictionary
        candidate_pool = [w for w in candidate_pool if w not in self.namedict]
        # remove the words that contains digits
        alphab_words = []
        for w in candidate_pool:
            if all(c.isalpha() for c in w):
                alphab_words.append(w)
        candidate_count = Counter(alphab_words)
        return [k for k,v in candidate_count.items() if v>thres]

class Pfidf():
    def __init__(self,texts,wordlist):
        '''
        Prepare the parameters to compute pfidf score
        cpf(w): the number of time that w is predicted as a name by the extractor
        ctf(w): number of occurences of the word w in the entire test corpus 
        pf = cpf/ctf
        df(w): the number of documents that contains a word w
        N: the number of documents in the corpus
        '''
        # preprocess the texts, tokenize
        self.N = len(texts)
        self.cleaned_text = [t.lower() for t in texts]
        nlp = English()
        self.tokenizer = nlp.tokenizer
        whole_tokens = []
        for doc in self.tokenizer.pipe(texts, batch_size=50):
            whole_tokens.extend([token.text.lower() for token in doc])
        print('first 20 whole tokens in pfidf:', whole_tokens[:20])
        
        # prepare the nltk.text
        self.text = nltk.Text(whole_tokens)
        # prepare the flattened and lower cased wordlist
        flatten_wl = []
        for n in wordlist:
            # for w in n:
            if all(c.isalpha() for c in n):
                flatten_wl.append(n.lower())
        # compute the pfidf of each word in wordlist
        self.pfidf_dict = self.pf_idf(flatten_wl)
        
    def expand(self, thres=0.2, verbose=0):
        if verbose:
            for k,v in self.pfidf_dict.items():
                if v>=thres:
                    print('='*120)
                    print(k,':',v)
                    print(self.text.concordance(k))
                    print()
        return [k for k,v in self.pfidf_dict.items() if v>=thres]
    
    def pf_idf(self,extracted_names):
        cpf_dict = self.compute_cpf_dict(extracted_names)
        ctf_dict = self.compute_ctf_dict(list(cpf_dict.keys()))
        pf_dict = self.compute_pf_dict(cpf_dict, ctf_dict)

        pfidf_dict = {key : 0 for key in cpf_dict.keys()}
        for word in cpf_dict.keys():
            pf = pf_dict[word]
            idf = self.compute_idf(word)
            pfidf_dict[word] = math.exp(pf) * idf
        return pf_dict
    
    def compute_cpf_dict(self,extracted_names):
        result = {}
        for name in extracted_names:
            if name not in result:
                result[name]=extracted_names.count(name)

        return result
    
    def compute_ctf_dict(self, unique_names):
        result = {}
        for name in unique_names:
            result[name] = self.text.count(name)
        return result

    def compute_pf_dict(self,cpf_dict, ctf_dict):
        '''Predicted Frequency:
        estimates the degree to which a word appears to be used consistently as a name throughout the corpus'''
        pf_dict = {key : 0 for key in cpf_dict.keys()}
        for word,cpf in cpf_dict.items():
            if ctf_dict[word] >0:
                pf_dict[word] = cpf/ctf_dict[word]
            else:
                pf_dict[word] = -1
        return pf_dict


    def compute_df(self,word):
        return len([True for text in self.cleaned_text if word.lower() in text])

    def compute_idf(self,word):
        doc_num = self.compute_df(word)
        if doc_num == 0:
            return 0
        num = math.log((self.N+0.5) / doc_num)
        den = math.log(self.N+1)
        return num / den

def load_corpus(file):
    with open(file, 'r') as f:
        corpus = [line.strip() for line in f.readlines()]
    return corpus

def load_json(file):
    with open(file) as json_file:
        weights = json.load(json_file)
    return weights


def write_json(weighted_dict, output):
    with open(output, 'w') as json_file:
        json.dump(weighted_dict, json_file)

def main():
    parser = argparse.ArgumentParser(description='to do')
    parser.add_argument('-i', type=str, help='filepath of the csv file')
    parser.add_argument('-o', type=str, help='output file path of the expanded weighted dictionary')
    parser.add_argument('-d', type=str, help='file path of the current weighted dictionary')
    args = parser.parse_args()

    corpus = load_corpus(args.i)
    namelist = load_json(args.d)

    cleaned_corpus = [preprocess(cps) for cps in corpus]
    rule_ex = RuleExtractor()
    rule_names = extract_names(cleaned_corpus, rule_ex)
    rule_names = [name.text for names in rule_names for name in names]
    full_ex = NameExtractor(weights_dict=args.d)
    full_names= extract_names(cleaned_corpus, full_ex)

    pfidf_expand = Pfidf(cleaned_corpus, rule_names)
    extractor_expand = EXT(full_names)



    # candidates = expand_dict(args.f)
    # print(candidates)


# def expand_dict(filepath):
#     """
#     Return a list of (word, confidence_score) pair.
#     1. load dataset, clean the corpus
#     2. extract names from corpus using dict and rule separately
#     3. apply pfidf and word2vec

#     Input: filepath of the csv file 
#     """
#     corpus, namelist = load_data(filepath)
#     cleaned_corpus = [preprocess(cps) for cps in corpus]
#     dict_names, regex_names = extract_names(cleaned_corpus)
#     candidates = calculate_candidate(corpus, namelist, dict_names, regex_names)

#     return candidates

def extract_names(corpus, extractor):
    names = []

    for text in corpus:
        names.append([ent.text for ent in extractor.extract(text)])

    return names

def calculate_candidate(corpus, namelist,dict_names, regex_names):
    # initialize pfidf and word2vec instance
    pfidf_expand = Pfidf(corpus,regex_names)
    w2v_expand =W2V(corpus,dict_names, namelist)

    pfidf_candidates = pfidf_expand.expand(thres=0.2, verbose=0)
    w2v_candidates = w2v_expand.expand(thres=5)
    print('pfidf_candidates:', pfidf_candidates)
    print('word2vec candidates:', w2v_candidates)

    total_candidates = set([w.lower() for w in pfidf_candidates]) | set([w.lower() for w in w2v_candidates])
    return list(total_candidates)


if __name__ == '__main__':
    main()


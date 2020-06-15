from extractors.extractor import Extractor
import joblib 
import spacy
import pandas as pd

class CRFExtractor(Extractor):
    def __init__(self, extractor_name=None, dict_file='extractors/src/nameslist.csv'):
        Extractor.__init__(self, name=extractor_name)
        self.model = joblib.load('extractors/src/crf_model.joblib') 
        self.nlp = spacy.load("en_core_web_sm")
        self.dictionary = self.load_word_dict(dict_file)


    def load_word_dict(self,dict_file):   
        df_names = pd.read_csv(dict_file)
        df_names.drop_duplicates(subset='Name', inplace=True)
        df_names.dropna()
        newwordlist = df_names['Name']
        return [word.strip().lower() for word in newwordlist]

    def extract(self, text):
        feature_list = self.txt2features(text)
        tag_list = self.model.predict_single(feature_list)
        word_list = self.txt2words(text)
        
        result = []
        for word, tag in zip(word_list, tag_list):
            if tag == 'PERSON':
                result.append(word[0].lower())
        return result

    
    def txt2features(self, text):
        sent = self.txt2words(text)
        sent_features = [self.sent2features(sent, i) for i in range(len(sent))]
        return sent_features


    def txt2words(self, text):
        str_tokens = self.nlp(text)
        return [(token.text,token.pos_) for token in str_tokens]

    def sent2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 0.02,
            #'word.lower()': word.lower(),
            'word[-2:]': word[-2:],
            'word[-1:]': word[-1:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
            'inNameList': word.lower() in self.dictionary,
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features
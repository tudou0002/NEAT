from allennlp_models.pretrained import load_predictor
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

file = sys.argv[1]
data = pd.read_csv(file, sep='\t')

text = data.description.values
results = []
all_words = []

predictor = load_predictor("tagging-elmo-crf-tagger")

for sent in tqdm(text):
	predicted = []
	res = predictor.predict(sent)
	words, preds = res['words'], res['tags']
	for tag in preds:
		# words.append(sent.split()[i])
		if len(tag) == 1:
			predicted.append('0')
		else:
			# predicted.append(tag[1])
			predicted.append(tag.split('-')[1])
	results.append(predicted)
	all_words.append(words)
data['elmo'] = results
data.to_csv(file, sep='\t', index=False)
print(data.head())


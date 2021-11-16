from allennlp_models.pretrained import load_predictor
from tqdm import tqdm
import pandas as pd
import sys

file = sys.argv[1]
data = pd.read_csv(file, sep='\t')
col = sys.argv[2]

# text = data.description.values
text = data[col].values
results = []

predictor = load_predictor("tagging-elmo-crf-tagger")

for sent in tqdm(text):
	names_per_ad = []
	preds = predictor.predict(sent)

	for word, tag in zip(preds["words"], preds["tags"]):
		if 'PER' in tag:
			# print(word)
			names_per_ad.append(word)
	results.append(names_per_ad)


data['elmo_names'] = results
print(data.head())
data.to_csv(file,sep='\t',index=False)

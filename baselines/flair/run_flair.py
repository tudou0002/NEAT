import sys
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

file = sys.argv[1]
col = sys.argv[2]

data = pd.read_csv(file, sep='\t')
# text = data.description.values
text = data[col].values

# # load the NER tagger
tagger = SequenceTagger.load('ner')
results = []

for sent in tqdm(text):
	names_in_ad = []
	sentence = Sentence(sent)
	# run NER over sentence
	tagger.predict(sentence)

	# iterate over entities and print
	entities = sentence.to_dict(tag_type='ner')['entities']
	for entity in entities:
		label = str(entity['labels'][0]).split()[0]
		# print(label, entity['text'])
		if 'PER' in label:
			word = entity['text'].lower()
			names_in_ad.extend(word.split())
	results.append(names_in_ad)

print(results)
data['flair_names'] = results
data.to_csv(file,sep='\t',index=False)



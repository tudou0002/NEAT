import sys
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

file = sys.argv[1]

data = pd.read_csv(file, sep='\t')
text = data.description.values

# # load the NER tagger
tagger = SequenceTagger.load('ner')
# tagger = SequenceTagger.load("flair/ner-english-fast")
results = []

for sent in tqdm(text[:10]):
	names_in_ad = []
	sentence = Sentence(sent)
	# run NER over sentence
	tagger.predict(sentence)

	# iterate over entities and print
	entities = sentence.to_dict(tag_type='ner')['entities']
	for entity in entities:
		label = str(entity['labels'][0]).split()[0]
		if 'PER' in label:
			names_in_ad.append(entity['text'].lower())
	results.append(names_in_ad)

data['flair'] = results
data.to_csv(file,sep='\t',index=False)
print(data.head())


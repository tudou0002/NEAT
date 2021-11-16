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
all_words = []

for sent in tqdm(text):
	sentence = Sentence(sent)
	# run NER over sentence
	tagger.predict(sentence)
	# iterate over entities and print
	entities = sentence.to_dict(tag_type='ner')['entities']
	ents = []
	words = []
	for word in sent.split():
		word_flag = False
		for entity in entities:
			if entity['text'] == word:
				# label = str(entity['labels'][0])
				label = str(entity['labels'][0]).split()[0]
				ents.append(label)
				word_flag = True
				break
		if not word_flag:
			ents.append('0')
		words.append(word)
	results.append(ents)
	all_words.append(words)

data['flair_ner_tags'] = results
data['flair_ner_words'] = all_words
data.to_csv(file,sep='\t',index=False)
print(data.head())


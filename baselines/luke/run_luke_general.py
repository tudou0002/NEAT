from transformers import LukeTokenizer, LukeForEntitySpanClassification
import pandas as pd 
from tqdm import tqdm
import numpy as np
import sys

from sklearn.metrics import precision_score, recall_score, f1_score

MAX_SENT_LEN = 512

file = sys.argv[1]
data = pd.read_csv(file,sep='\t')
texts = data.description.values

model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

# text = "Beyoncé lives in Los Angeles."
# text = data.description.values

def get_entity_spans(text):
# function returns the entity spans in a sentence
	word_start = []
	word_end = []
	word_has_started = False

	for i, char in enumerate(text):
		if char != ' ' and not word_has_started:
			word_has_started = True
			word_start.append(i)
		if char == ' ' and word_has_started:
			word_end.append(i)
			word_has_started = False

	word_end.append(len(text)-1)

	entity_spans = []
	for i, start_pos in enumerate(word_start):
		end_pos = word_end[i]
		# for end_pos in word_end[i:]:
		entity_spans.append(tuple([start_pos, end_pos]))

	return entity_spans

results = []
all_words = []
skip = 0
label_map = {'0':0, 'B-LOC':1, 'B-MISC':2, 'B-ORG':3, 'B-PER':4, 'I-LOC':5, 'I-MISC':6, 'I-ORG':7, 'I-PER':8}

for sent in tqdm(texts):
	ent_spans = get_entity_spans(sent)
	inputs = tokenizer(sent, entity_spans=ent_spans, return_tensors="pt")
	# print(inputs)
	outputs = model(**inputs)
	logits = outputs.logits
	predicted_class_indices = logits.argmax(-1).squeeze().tolist()
	preds = []
	words = []
	if type(predicted_class_indices) == int:
		p = model.config.id2label[predicted_class_indices]
		# print(p)
		preds.append(p)
		results.append(preds)
		words.append(sent[span[0]:span[1]])
		all_words.append(words)
		continue
	for span, pred in zip(ent_spans, predicted_class_indices):
		if pred != 0:
			pred_label = model.config.id2label[pred]
			preds.append(pred_label)
		else:
			preds.append('0')
		words.append(sent[span[0]:span[1]])
		# print(sent[span[0]:span[1]])
	results.append(preds)
	all_words.append(words)


data['luke_ner_tags'] = results
data['luke_ner_words'] = all_words
data.to_csv(file, sep='\t',index=False)
print(data.head())


# with open(file,'r') as f:
# 	lines = f.readlines()
# 	txt = ""
# 	true_labels = []
# 	pred_labels = []
# 	for line in tqdm(lines):
# 		if line == '\n':
# 			continue
# 		else:
# 			token = line.split()[0]
# 			label = line.split()[1]
# 			true_labels.append(label_map[label])
# 			inputs = tokenizer(token, entity_spans=[(0,1)], return_tensors="pt")

# 			outputs = model(**inputs)
# 			logits = outputs.logits
# 			predicted_class_indices = logits.argmax(-1).squeeze().tolist()
# 			if predicted_class_indices != 0:
# 				pred_label = model.config.id2label[predicted_class_indices]
# 			else:
# 				pred_label = '0'
# 			if pred_label not in label_map.keys():
# 				pred_label = 'I-' + pred_label
# 				print(pred_label)
# 			pred_labels.append(label_map[pred_label])

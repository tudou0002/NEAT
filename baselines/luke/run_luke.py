from transformers import LukeTokenizer, LukeForEntitySpanClassification
import pandas as pd 
from tqdm import tqdm
import numpy as np
import sys

MAX_SENT_LEN = 512

file = sys.argv[1]
data = pd.read_csv(file,sep='\t')

model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

# text = "Beyoncé lives in Los Angeles."
text = data.description.values

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
skip = 0
for sent in tqdm(text):
	names_per_ad = []
	entity_spans = get_entity_spans(sent)
	inputs = tokenizer(sent, entity_spans=entity_spans, return_tensors="pt")
	if len(inputs['input_ids'][0]) > MAX_SENT_LEN:
		skip += 1
		results.append([])
		continue
	outputs = model(**inputs)
	logits = outputs.logits
	predicted_class_indices = logits.argmax(-1).squeeze().tolist()
	if type(predicted_class_indices) == int:
		print(sent)
		results.append([])
		continue
	for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
		if predicted_class_idx != 0:
			label = model.config.id2label[predicted_class_idx]
			if 'PER' in label:
				name = sent[span[0]:span[1]]
				names_per_ad.append(name)
	# if len(sent) <= 3:
	# 	continue
	# if len(sent) > 500:
	# 	sents = [sent[:500], sent[500:]]
	# else:
	# 	sents = [sent]
	# for s in sents:
	# 	entity_spans = get_entity_spans(s)
	# 	# print(len(entity_spans))
	# 	if len(entity_spans) > 5000:
	# 		n = int(np.floor(len(entity_spans)/5000))
	# 		es = []
	# 		for i in range(n):
	# 			es.append(entity_spans[5000*i:5000*(i+1)])
	# 		es.append(entity_spans[5000*n:])
	# 		# es = [entity_spans[:5000], entity_spans[5000:]]
	# 	else:
	# 		es = [entity_spans]

	# 	for esp in es:
	# 		print(len(esp))
	# 		inputs = tokenizer(s, entity_spans=esp, return_tensors="pt")
	# 		# print(len(inputs), s)
	# 		outputs = model(**inputs)
	# 		logits = outputs.logits
	# 		predicted_class_indices = logits.argmax(-1).squeeze().tolist()
	# 		if type(predicted_class_indices) == int:
	# 			print(sent)
	# 			continue
	# 		for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
	# 			if predicted_class_idx != 0:
	# 				label = model.config.id2label[predicted_class_idx]
	# 				if 'PER' in label:
	# 					name = s[span[0]:span[1]]
	# 					names_per_ad.append(name)
	results.append(names_per_ad)

data['luke'] = results
print(data.head())
data.to_csv(file, sep='\t')

print("Skip = " + str(skip))
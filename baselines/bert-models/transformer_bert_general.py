import sys
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

# from datasets import load_dataset # this has to be installed first using - !pip install datasets
# transformers package can be installed using "!pip install transformers"
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification

from transformers import BertTokenizer

import re, unicodedata, os, logging, unidecode, emoji
from html.parser import HTMLParser


label_map = {'0':0, 'B-LOC':1, 'B-MISC':2, 'B-ORG':3, 'B-PER':4, 'I-LOC':5, 'I-MISC':6, 'I-ORG':7, 'I-PER':8, 'O':0}


def get_args():
	print("Getting arguments .... \n")
	args = argparse.ArgumentParser()
	args.add_argument('--dataset', default='conll', help='enter the path of the csv datafile with ads/text')
	args.add_argument('--model', default='transformer-bert', help='enter the path to the baseline model you want to use such as \
		fine-tuned-bert,ht-bert,whole-mask-bert. For transformer-bert, simply type "transformer-bert".')
	args.add_argument('--res_column_name', default='transformer-bert')
	args.add_argument('--preprocess', default=True)
	return args.parse_args()

def character_ends(name):
	# for those predictions with character ##
	new_names = []
	new_name = ""
	for i in range(len(name)):
		if '##' in name[i]:
			new_name += name[i][2:]
		else:
			if new_name != "":
				new_names.append(new_name)
				new_name = ""
			new_name += name[i]
	if new_name != '':
		new_names.append(new_name)
	return new_names

def character_starts(name):
	new_names = []
	new_name = ""
	for i in range(len(name)):
		if chr(288) in name[i]:
			if new_name != "" :
				# print(new_name)
				new_names.append(new_name)
				new_name = ""
			new_name += name[i][1:]
		else:
			new_name += name[i]
	if new_name != '':
		new_names.append(new_name)
	return new_names

def post_process(names):
	# this function converts the bert-tokenized words into detokenized simple string
	# input: list of names in bert tokenized format
	# output: list of names detokenized
	print("Detokenizing ..... \n")
	# joining together names which got tokenized with the # character
	all_new_names = []

	for name in names:
		new_names = []
		if len(name) == 1:
			all_new_names.append(name)
			continue
		j = 0
		new_name = ""
		if chr(288) in ''.join(name): # some models return predictions with a special character (ascii code 288)
			new_names = character_starts(name)
		elif '##' in ''.join(name):
			new_names = character_ends(name)

		all_new_names.append(new_names) 

	# modifying the names to remove special characters
	mod_names = []
	for n in all_new_names:
		tmp_names = []
		for name in n:
			name = name.replace("</s>", "")
			name = name.replace("#", "")
			name = name.replace("<s>", "")
			name = name.replace(" ", "")
			name = name.encode('ascii',errors='ignore')
			name = name.decode()
			if name != ' ' or name != '':
				tmp_names.append(name.lower())
		mod_names.append(tmp_names)

	return mod_names 


def get_model(model_name):
	# this function uses bert pipeline for NER
	'''
	input: list of text/ads
	output: list of names extracted in the same order as the input list
	'''
	# print("Extracting names ..... \n")

	if model_name == 'transformer-bert':
		transformer_entity_extractor = pipeline("ner")
		model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
		tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
	else:
		model = AutoModelForTokenClassification.from_pretrained(Path(model_name))
		tokenizer = AutoTokenizer.from_pretrained(Path(model_name))

		transformer_entity_extractor = pipeline('ner', model=model, tokenizer=tokenizer)
	
	return transformer_entity_extractor, tokenizer, model


def main():
	args = get_args()
	_, tokenizer, model = get_model(args.model)

	data_file = args.dataset
	results = []
	all_words = []
	print("Loading data ..... \n")

	data = pd.read_csv(data_file, sep='\t')
	texts = data.description.values
	for text in tqdm(texts):
		tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
		inputs = tokenizer.encode(text, return_tensors="pt")
		outputs = model(inputs).logits
		predictions = torch.argmax(outputs, dim=2)
		sent_res = []
		words = []
		word = ""
		labs = []
		for token, prediction in zip(tokens, predictions[0].numpy()):
			# print(token, prediction)
			if token[0] == '#':
				word += token[2:]
				labs.append(prediction)
			else:
				if word == "":
					word += token
					labs.append(prediction)
				else:
					if word not in ['[CLS]','[SEP]']:
						pred = list(set(labs))[0]
						pred = model.config.id2label[pred]
						if len(pred) == 1:
							pred = '0'
						# print(word, pred)
						words.append(word)
						sent_res.append(pred)

					word = token
					labs = [prediction]
		results.append(sent_res)
		all_words.append(words)

	all_words = post_process(all_words)
	data[args.res_column_name+"_tag"] = results
	data[args.res_column_name+"_words"] = all_words
	print(data.head())
	data.to_csv(data_file, sep='\t',index=False)

main()

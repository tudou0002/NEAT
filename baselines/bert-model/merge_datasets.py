# This script merges the formatted custom datasets with the ConLL 2003 dataset for fine-tuning bert model for NER

import argparse
from datasets import load_dataset
import ast
import pandas as pd
import numpy as np

from tqdm import tqdm

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM, RobertaForTokenClassification

import os
import re
import logging
import unicodedata
from html.parser import HTMLParser
import emoji
import unidecode

tr_size = 5000
test_size = 1000

# compile regexes
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
control_char_regex = re.compile(r'[\r\n\t]+')
# translate table for punctuation
transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-",  u"'''\"\"--")])
# HTML parser
html_parser = HTMLParser()

def string_to_list(input):
	# function to convert a string of a list to a list
	if type(input) == list or input is None:
		return ['']
	new_list = []
	for item in input.split(','):
		if item == 'set()':
			new_list.append('')
			continue
		elif item[0] == '[':
			if item[1] == "'":
				it = item[2:-1]
			else:
				it = item[1:]
		elif item[-1] == ']':
			if item[0] == ' ':
				it = item[1:-1]
			else:
				it = item[:-1]
			if it[0] == "'":
				it = it[1:-1]
		elif item[0] == ' ':
			it = item[1:]
			if it[0] == "'" or it[0] == '"':
					it = it[1:-1]
			else:
					it = it[:]
		else:
			it = item[1:-1]
			if it[0] == "'" or it[0] == '"':
					it = it[1:-1]
			else:
					it = it[:]
		new_list.append(it)
	return new_list

def apply_mapping(x):
	tag_correction_map = {'0':'0', 0:'0', 'I-PER':'I-PER','B-PER':'B-PER',1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', \
	5:'B-LOC',6:"I-LOC",7:"B-MISC",8:"I-MISC", '':'0'}

	# if x != 0 and x != '0' and x != 1 and x != 2 and x !='I-PER' and x != 'B-PER':
	# 	print(x)
	# 	#count += 1

	return tag_correction_map[x]

def merge(conll, custom_train, custom_test, conll_flag=False):
	# merges the custom HT dataset with the conll for downstream NER
	# also saves the test sets for conll, the HT data (listcrawler) and the full conll for the sanity-check experiments
	if conll_flag:
		custom_train = pd.DataFrame(columns=['tokens','ner_tags'])
		custom_test = pd.DataFrame(columns=['tokens','ner_tags'])

	for id, item in tqdm(enumerate(conll['train']['tokens'][:tr_size])):
		custom_train = custom_train.append({'tokens':item, 'ner_tags':conll['train']['ner_tags'][id]}, ignore_index=True)

	for id, item in tqdm(enumerate(conll['test']['tokens'][:test_size])):
		custom_test = custom_test.append({'tokens':item, 'ner_tags':conll['test']['ner_tags'][id]}, ignore_index=True)

	print(custom_train.shape)
	print(custom_test.shape)

	print(custom_train.ner_tags.value_counts())

	return custom_train, custom_test


def load_conll():
	# returns the conll dataset
	conll = load_dataset('conll2003')
	new_conll = {'train':{'tokens':[None]*tr_size, 'ner_tags':[None]*tr_size}, 'test':{'tokens':[None]*test_size, 'ner_tags':[None]*test_size}}
	# removing class labels other than B-PER and I-PER
	for id, item in tqdm(enumerate(conll['train']['ner_tags'][:tr_size])):
		new_list = []
		new_conll['train']['tokens'][id] = conll['train']['tokens'][id]
		new_conll['train']['ner_tags'][id] = item

	for id, item in tqdm(enumerate(conll['test']['ner_tags'][:test_size])):
		new_list = []
		new_conll['test']['tokens'][id] = conll['test']['tokens'][id]
		new_conll['test']['tokens'][id] = item
	return new_conll


def preprocess_bert(text, do_lower_case=False):
	"""Preprocesses input for NER"""
	# standardize
	text = standardize_text(text)
	text = asciify_emojis(text)
	text = standardize_punctuation(text)
	if do_lower_case:
		text = text.lower()
	text = remove_unicode_symbols(text)
	text = remove_accented_characters(text)
	return text

def remove_accented_characters(text):
	text = unidecode.unidecode(text)
	return text

def remove_unicode_symbols(text):
	text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')
	return text

def asciify_emojis(text):
	"""
	Converts emojis into text aliases. E.g. 👍 becomes :thumbs_up:
	For a full list of text aliases see: https://www.webfx.com/tools/emoji-cheat-sheet/
	"""
	text = emoji.demojize(text)
	return text

def standardize_text(text):
	"""
	1) Escape HTML
	2) Replaces some non-standard punctuation with standard versions. 
	3) Replace \r, \n and \t with white spaces
	4) Removes all other control characters and the NULL byte
	5) Removes duplicate white spaces
	"""
	# escape HTML symbols
	text = html_parser.unescape(text)
	# standardize punctuation
	text = text.translate(transl_table)
	text = text.replace('…', '...')
	# replace \t, \n and \r characters by a whitespace
	text = re.sub(control_char_regex, ' ', text)
	# remove all remaining control characters
	text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
	# replace multiple spaces with single space
	text = ' '.join(text.split())
	return text.strip()

def standardize_punctuation(text):
	return ''.join([unidecode.unidecode(t) if unicodedata.category(t)[0] == 'P' else t for t in text])

def is_word_in_list(word, list_of_names):
	if word in list_of_names:
		return True
	for name in list_of_names:
		if (word in name.lower()) and len(word) > 2:
			return True
	return False


def get_tags_in_format(list_of_text, list_of_names, filename):
	all_words = []
	all_tags = []
	all_ids = []
	c = 0
	list_of_names = [list(map(lambda x: x.lower(), names)) for names in list_of_names]
	check = {}
	for id, text in enumerate(list_of_text):
		split_by_comma = text.split(',') # we have to do this as some names appear just before a comma without /
										# spaces and we will miss those
		words_in_text = []
		tags = []
		all_ids.append(str(id))
		for txt in split_by_comma: # splitting again by spaces. So essentially splitting by both comma & space
			words_in_text.extend(txt.split())

		text = text.replace(',',' ')
		text = text.replace('.',' ')
		text = text.replace('/',' ')
		text = text.replace('*',' ')
		text = text.replace('(',' ')
		text = text.replace(')',' ')
		text = text.replace(':',' ')
		text = text.replace('-',' ')
		text = text.replace('!',' ')

		words_in_text = text.split()

		if len(list_of_names[id]) == 1 and (list_of_names[id][0] == '' or list_of_names[id][0] == ']'):
			check[id] = True

		for word in words_in_text:
			w = word.lower().strip()
			if len(list_of_names[id]) == 1 and (list_of_names[id][0] == '' or list_of_names[id][0] == ']'):
				tag = 0
			elif is_word_in_list(w, list_of_names[id]):
				if w in words_in_text[:3]: # tag is B-PER
					tag = "B-PER"
				else:
					tag = "I-PER"
				check[id] = True
			else:
				tag = 0
			tags.append(tag)
		
		all_words.append(words_in_text)
		all_tags.append(tags)

	new_data = {'id':all_ids, 'tokens':all_words, 'ner_tags':all_tags}
	df = pd.DataFrame(new_data)
	df.to_csv(filename)
	print("Count = " + str(len(check)))
	return df

def load_custom_data(train_data_file, test_data_file, train_desc_column, test_desc_column, train_tag_column, test_tag_column, split):

	train_data = pd.read_csv(train_data_file, sep='\t')
	# ht_data_for_testing = train_data.iloc[200:]
	# ht_data_for_testing.to_csv("../../data/Listcrawler_baseline_test.tsv",sep='\t')
	# train_data = train_data.iloc[:200]
	train_data['processed_text'] = train_data[train_desc_column].apply(lambda x: preprocess_bert(x))
	train_data[train_tag_column] = train_data[train_tag_column].apply(lambda x: string_to_list(x))
	new_data_train = get_tags_in_format(train_data['processed_text'], train_data[train_tag_column], "/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_for_fine_tune_jun9/ht_formatted_data_train.csv")

	if split:
		test_data = pd.read_csv(test_data_file, sep='\t')
		test_data['processed_text'] = test_data[test_desc_column].apply(lambda x: preprocess_bert(x))
		test_data[test_tag_column] = test_data[test_tag_column].apply(lambda x: string_to_list(x))
		new_data_test = get_tags_in_format(test_data['processed_text'], test_data[test_tag_column], "/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_for_fine_tune_jun9/ht_formatted_data_test.csv")
	
		return new_data_train, new_data_test

	return new_data_train, None

def csv_to_txt(train, test, train_path, test_path, split):

	# new_train = open("/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/test_sets/train.txt", "w")
	new_train = open(train_path, 'w')
	print(train.shape)

	if split:
		# new_test = open("/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_feb18/test.txt", "w")
		new_test = open(test_path, 'w')
		print(test.shape)

	# id_to_tag_map = {'0':'0', '1':'B-PER', '2':'I-PER'}	

	for id, row in tqdm(train.iterrows()):
		tokens = row['tokens']
		tags = row['ner_tags']

		for i, tok in enumerate(tokens):
			if len(tok) == 0:
				continue
			try:
				tag = apply_mapping(tags[i])
			except:
				tag = apply_mapping(tags[i][:-1])
			if ',' in tok or ',' in tag:
				print(tok)
				print(tag)
			else:
				new_train.write(tok + " " + str(tag) + "\n")
		new_train.write("\n")
	new_train.close()

	if split:
		for id, row in tqdm(test.iterrows()):
			tokens = row['tokens']
			tags = row['ner_tags']

			for i, tok in enumerate(tokens):
				if tags is None:
					continue
				tag = apply_mapping(tags[i])
				if ',' in tok or ',' in tag:
					print(tok)
					print(tag)
				else:
					new_test.write(str(tok) + " " + str(tag) + "\n")
			new_test.write("\n")

		new_test.close()

	#train = pd.read_csv("/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_feb18/train.txt")
	#test = pd.read_csv("/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_feb18/test.txt")

	#train.to_csv("/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_feb18/train.csv", index=False)
	#test.to_csv("/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_feb18/test.csv", index=False)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='to do')
	parser.add_argument('-train_data', type=str, default='../../data/Listcrawler_baseline.tsv', help='filepath of your custom train dataset as csv or tsv')
	parser.add_argument('-test_data', type=str, default='../../data/CanadaMax80_results.csv', help='filepath of your custom test dataset as csv or tsv')

	parser.add_argument('-train_dc', type=str, default='description', help='name of the description column of train data')
	parser.add_argument('-train_tc', type=str, default='MTurk_1', help='name of the column of ner tags in required format of train data')

	parser.add_argument('-test_dc', type=str, default='description', help='name of the description column of test data')
	parser.add_argument('-test_tc', type=str, default='TRUE', help='name of the column of ner tags in required format of test data')

	parser.add_argument('--merge', action="store_true", help='indicates whether to merge with conll or not')
	parser.add_argument('--split', action='store_true', help='split into train-test or not')
	parser.add_argument('--conll', action='store_true', help='indicates whether to process only conll or not')

	args = parser.parse_args()
	
	if args.conll:
		conll = load_conll()
		print("CoNLL loaded ...\n")
		conll_train, conll_test = merge(conll, None, None, conll_flag=True)
		print("Train-test split achieved ...\n")
		csv_to_txt(conll_train, conll_test, "/home/mcb/users/pnair6/NER/NameExtractor/data/conll_only_train.txt","/home/mcb/users/pnair6/NER/NameExtractor/data/conll_only_test.txt", args.split)

	elif args.merge:
		conll = load_conll()
		data_train, data_test = load_custom_data(args.train_data, args.test_data, args.train_dc, args.test_dc, args.train_tc, args.test_tc, args.split)
		# "/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_feb18/ht_formatted_data"
		new_train, new_test = merge(conll, data_train, data_test)
		train_path = "/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_for_fine_tune_jun2/train.txt"
		test_path = "/home/pnair6/McGill/Research/HT/NER/bert_ner_fine_tune/data_for_fine_tune_jun2/test.txt"
		csv_to_txt(new_train, new_test, train_path, test_path, args.split)		
	else:
		new_train, new_test = data_train, data_test

		train_path = args.train_data[:-3]+"txt"

		if args.split:
			test_path = args.test_data[:-3]+"txt"
		else:
			test_path = ""

		csv_to_txt(new_train, new_test, train_path, test_path, args.split)


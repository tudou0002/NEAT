import sys
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

# from datasets import load_dataset # this has to be installed first using - !pip install datasets
# transformers package can be installed using "!pip install transformers"
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification

from transformers import BertTokenizer

import re, unicodedata, os, logging, unidecode, emoji
from html.parser import HTMLParser

EMOJI_PATTERN = re.compile(
	"["
	"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	"\U0001F300-\U0001F5FF"  # symbols & pictographs
	"\U0001F600-\U0001F64F"  # emoticons
	"\U0001F680-\U0001F6FF"  # transport & map symbols
	"\U0001F700-\U0001F77F"  # alchemical symbols
	"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
	"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
	"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
	"\U0001FA00-\U0001FA6F"  # Chess Symbols
	"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
	"\U00002702-\U000027B0"  # Dingbats
	"\U000024C2-\U0001F251" 
	"]+"
)

# compile regexes
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
control_char_regex = re.compile(r'[\r\n\t]+')
# translate table for punctuation
transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-",  u"'''\"\"--")])
# HTML parser
html_parser = HTMLParser()

def basic_preprocess(text):
	# return empty string if text is NaN
	if type(text)==float:
		return ''
	# remove emoji
	text = re.sub(EMOJI_PATTERN, r' ', text)
	text = re.sub(r'·', ' ', text)
	# convert non-ASCII characters to utf-8
	text = unicodedata.normalize('NFKD',text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	return text

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

def get_args():
	print("Getting arguments .... \n")
	args = argparse.ArgumentParser()
	args.add_argument('--dataset', default='conll', help='enter the path of the csv datafile with ads/text')
	args.add_argument('--model', default='transformer-bert', help='enter the path to the baseline model you want to use such as \
		fine-tuned-bert,ht-bert,whole-mask-bert. For transformer-bert, simply type "transformer-bert".')
	args.add_argument('--inp_column_name', default='description')
	args.add_argument('--res_column_name', default='transformer-bert')
	args.add_argument('--preprocess', default=True)
	return args.parse_args()

def detokenize(names):
	# this function converts the bert-tokenized words into detokenized simple string
	# input: list of names in bert tokenized format
	# output: list of names detokenized
	print("Detokenizing ..... \n")
	def is_subtoken(word):
		if word[:2] == "##":
			return True
		else:
			return False

	new_names = []
	for j, tokens in enumerate(names):
		restored_text = []
		for i in range(len(tokens)):
			if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
				restored_text.append(tokens[i] + tokens[i+1][2:])
				if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
					restored_text[-1] = restored_text[-1] + tokens[i+2][2:]

				if (i+3)<len(tokens) and is_subtoken(tokens[i+3]):
					restored_text[-1] = restored_text[-1] + tokens[i+3][2:]

			elif not is_subtoken(tokens[i]):
				restored_text.append(tokens[i])
		new_names.append(restored_text)

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
	    for i in range(len(name)-1):
	        new_name = name[j]
	        k = 1
	        while '#' in name[j+k]:
	            temp_name = name[j+k].replace('#',"")
	            new_name += temp_name
	            k += 1
	            if (j+k) >= len(name):
	                break
	        new_names.append(new_name)
	        if (j+k) >= len(name)-1:
	            break
	        else:
	            j += k
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


def extract_names(text, model_name):
	# this function uses bert pipeline for NER
	'''
	input: list of text/ads
	output: list of names extracted in the same order as the input list
	'''
	print("Extracting names ..... \n")

	if model_name == 'transformer-bert':
		transformer_entity_extractor = pipeline("ner")
		# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
		# transformer_entity_extractor = pipeline("ner", tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased"))
	else:
		model = AutoModelForTokenClassification.from_pretrained(Path(model_name))
		tokenizer = AutoTokenizer.from_pretrained(Path(model_name))

		transformer_entity_extractor = pipeline('ner', model=model, tokenizer=tokenizer)
	
	
	name_extracted = []
	for i, example in tqdm(enumerate(text)):
		tt = transformer_entity_extractor(example)
		names = []
		for item in tt:
			if len(item.keys()) == 0:
				continue
			if 'PER' in item['entity']:
				names.append(item['word'])
		name_extracted.append(names)
	return post_process(name_extracted)

def save_data(args, data_df):
	# function writes the dataframe with the extracted names added as a column
	data_file = args.dataset
	if data_file.strip()[-3:] == 'tsv':
		delimiter = '\t'
	else:
		delimiter = ','
	data_df.to_csv(data_file, sep=delimiter)


def load_data(args):
	# loads the data file which is a csv/tsv file with a column named 'description' and preprocesses it
	data_file = args.dataset
	print("Loading data ..... \n")

	if data_file.strip()[-3:] == 'tsv':
		delimiter = '\t'
	else:
		delimiter = ','

	data_df = pd.read_csv(data_file, sep=delimiter)
	if args.inp_column_name not in data_df.columns:
		print("Input correct text column name in dataset \n")
		exit()

	if args.preprocess: # pre-processing the text 
		data_df[args.inp_column_name] = data_df[args.inp_column_name].apply(lambda x: basic_preprocess(x))
	return data_df

def main():
	args = get_args()
	data = load_data(args)

	extracted_names = extract_names(data[args.inp_column_name].values, args.model) # this function already does the detokenizing
	# cleaned_names = detokenize(extracted_names)
	data[args.res_column_name] = extracted_names
	save_data(args, data)
	print(data.head())

main()
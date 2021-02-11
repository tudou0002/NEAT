import sys
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from datasets import load_dataset # this has to be installed first using - !pip install datasets
# transformers package can be installed using "!pip install transformers"
from transformers import AutoTokenizer, pipeline

from transformers import BertTokenizer

import re, unicodedata

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

def get_args():
	print("Getting arguments .... \n")
	args = argparse.ArgumentParser()
	args.add_argument('--dataset', default='conll', help='enter the name of the csv datafile with ads/text. \
	 For benchmark datasets, simply give conll or wnut')
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

def extract_names(text):
	# this function uses bert pipeline for NER
	'''
	input: list of text/ads
	output: list of names extracted in the same order as the input list
	'''
	print("Extracting names ..... \n")
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
	transformer_entity_extractor = pipeline("ner", tokenizer = AutoTokenizer.from_pretrained("bert-base-cased"))
	
	name_extracted = []
	for i, example in tqdm(enumerate(text)):
		tt = transformer_entity_extractor(example)
		names = []
		for item in tt:
			if 'PER' in item['entity']:
				names.append(item['word'])
		name_extracted.append(names)
	return name_extracted

def save_data(args, data_df):
	# function writes the dataframe with the extracted names added as a column
	data_file = args.dataset
	if data_file.strip()[:-3] == 'tsv':
		delimiter = '\t'
	else:
		delimiter = ','
	data_df.to_csv(data_file, sep=delimiter)

def benchmark_data(dataset):
	# this function returns the dataset for conll and wnut
	return

def load_data(args):
	# loads the data file which is a csv/tsv file with a column named 'description' and preprocesses it
	data_file = args.dataset
	print("Loading data ..... \n")
	if data_file == 'conll' or data_file == 'wnut':
		return benchmark_data(data_file)
	if data_file.strip()[:-3] == 'tsv':
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
	extracted_names = extract_names(data[args.inp_column_name])
	cleaned_names = detokenize(extracted_names)
	data[args.res_column_name] = cleaned_names
	save_data(args, data)

main()
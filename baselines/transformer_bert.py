import sys
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from datasets import load_dataset # this has to be installed first using - !pip install datasets
# transformers package can be installed using "!pip install transformers"
from transformers import pipeline
from transformers import AutoTokenizer

from transformers import BertTokenizer

def get_args():
	args = argparse.ArgumentParser()
	args.add_argument('--dataset', default='conll')
	args.add_argument('--column_name', default='transformer-bert')
	args.add_argument('--preprocess', default=True)
	return args.parse_args()


def load_dataset(data_file):
	# loads the data file which is a csv/tsv file with a column named 'description'
	
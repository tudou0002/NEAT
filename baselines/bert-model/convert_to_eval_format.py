import pandas as pd
import argparse
import numpy as np


def get_names(file_path):
	# convert from 'I-PER/B-PER' form to extracted names
	file = open(file_path, 'r').read().split('\n')
	names = []
	name = []
	for f in file[:-1]:
		if f == '':
			names.append(name)
			name = []
			continue
		if 'PER' in f.split()[1]:
			name.append(f.split()[0])
	return names

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='to do')
	parser.add_argument('--file', type=str, help="file to be formatted")
	#parser.add_argument('--true_file', type=str, help="original data file to add the result")

	args = parser.parse_args()

	extracted_names = get_names(args.file)
	#print(extracted_names)
	print(len(extracted_names))
	np.save(args.file.split('/')[0]+"/ft_names.npy", extracted_names)

	# data = pd.read_csv(args.true_file, sep='\t', header=None, index_col=False)
	# data['ht-bert'] = extracted_names
	# print(data.head())
	#data.to_csv(args.true_file, sep='\t', index=False)
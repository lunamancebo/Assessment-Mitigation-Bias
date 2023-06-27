import re
import sys
import os


#URLS_RE = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b')
URLS_RE = re.compile(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*')

LISTING_RE = re.compile(r'^(|[a-z]?|[0-9]{0,3})(\-|\.)+( |\n)')

def remove_urls(text):
	return URLS_RE.sub('', text)

def replace_multi_whitespaces(line):
	return ' '.join(line.split())

def remove_listing(line):
	return LISTING_RE.sub('', line)

def get_labels(data,target,labels):
	final_labels = []
	for index,row in data.iterrows():
			target_label = row[target].replace(' ','')
			username = row['username']

			with open(f'../Cleaned Documents/{username}.txt', "r") as input_file:
				for line in input_file:
					if line == '\n':
						continue
					else:
						line = line.lower()
						line = remove_urls(line)
						line = remove_listing(line)
						line = replace_multi_whitespaces(line)

						if line != '':
							for label, val in labels:
								if label.replace('__label__','') == target_label:
									final_labels.append(val)

	return final_labels

def main(data,name,target=None,label=True):	

	file_path = f'{name}.txt'
	
	with open(file_path,'x') as f:
		for index,row in data.iterrows():
			target_label = row[target].replace(' ','')
			username = row['username']

			with open(f'../Cleaned Documents/{username}.txt', "r") as input_file:
				for line in input_file:
					if line == '\n':
						continue
					else:
						line = line.lower()
						line = remove_urls(line)
						line = remove_listing(line)
						line = replace_multi_whitespaces(line)

						if line != '':
							if label:
								f.write(f'{line} __label__{target_label}\n')
							else:
								f.write(f'{line}\n')
		
		return file_path

'''
Prepare dataset for stylistic and N-Gram features analysis by:
    - Removing mentions
    - Removing emojis 
    - Removing URL
    - Removing character '#'
'''

import pandas as pd
from emoji import demojize
import re

data = pd.read_excel('cleaned_users.xlsx')
username_list = list(data['username'])

for username in username_list:
    with open(f'Documents/{username}.txt','r') as f:
        text = f.read()

        # remove '#'
        text = text.replace('#','')
        # replace emojis with corresponding meaning
        text = demojize(text,delimiters=(' ',''),language='es')
        # remove mentions and url
        sentences = text.split('\n')
        words = []
        for sentence in sentences:
            words += sentence.split(' ')
        for word in words:
            if len(word) > 0:
                if '@' in word:
                    text = text.replace(word,'')
                elif 'http://' in word:
                    text = text.replace(word,'')
                elif 'https://' in word:
                    text = text.replace(word,'')

    #text = text.replace('\n\n','\n')
    # remove several whitespaces and breaklines concatenated
    aux = text.split(' ')
    text = " ".join([w for w in aux if len(w) > 0])
    aux = text.split('\n')
    aux_list = []
    for w in aux:
        if len(w) > 0:
            if w[0] == ' ':
                w = w.replace(' ','',1)
            aux_list.append(w)
    text = "\n".join(aux_list)
    with open(f'Cleaned Documents/{username}.txt','x') as f:
        f.write(text)
        
import re
import sys
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from emoji import emoji_count
import statistics as stat
import nltk
from nltk.corpus import cess_esp
import os
import json
import datetime
from dateutil.relativedelta import relativedelta
from collections import Counter

#URLS_RE = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b')
URLS_RE = re.compile(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*')

LISTING_RE = re.compile(r'^(|[a-z]?|[0-9]{0,3})(\-|\.)+( |\n)')

def remove_urls(text):
    return URLS_RE.sub('', text)

def replace_multi_whitespaces(line):
    return ' '.join(line.split())

def remove_listing(line):
    return LISTING_RE.sub('', line)

def remove_punctuation(text):
    text = text.replace('!','')
    text = text.replace('"','')
    return text.translate(str.maketrans('','',string.punctuation))

def quitar_guiones(text):
    aux = text.split('--')
    aux2 = []
    for w in aux:
        if len(w)>0:
            text = replace_multi_whitespaces(w)
            aux2.append(text)
    return '\n'.join(aux2)

def quitar_tags(text):
    tags = 'LOC|NUM|TIEM|PER'
    regex = re.compile(tags,re.S)
    res = regex.sub('', text)
    return res.replace('<','')

def remove_stopwords(text,stop_words):
    words = text.split(' ')
    not_stop_words = []
    for word in words:
        if word not in stop_words:
            not_stop_words.append(word)
    return ' '.join(not_stop_words)
      

def clean_text(text,stop_words):	

    low = text.lower()
    rem_u = remove_urls(low)
    rem_l = remove_listing(rem_u)
    rem_t = quitar_tags(rem_l)
    rem_w = replace_multi_whitespaces(rem_t)
    rem_p = remove_punctuation(rem_w)
    rem_s = remove_stopwords(rem_p,stop_words)
    text_enc = rem_s.encode('ascii', 'ignore')

    return text_enc.decode()

def tagger():
    oraciones = cess_esp.tagged_sents()
    return nltk.UnigramTagger(oraciones)


def load_embedding(file_path):
    embedding_vectors = {}
    with open(f'../../embeddings/{file_path}','r') as f:
        first_line = f.readline().split(' ')
        for line in f.readlines()[1:]:
            row = line.split(' ')
            word = row[0]
            if word not in embedding_vectors.keys():
                embedding_vectors[word] = [float(val) for val in row[1:]]
        
    return embedding_vectors

def load_sentiment_analysis():
    sent_analysis_data = pd.read_csv('../../Spanish-NRC-EmoLex.txt',sep='\t')
    spanish_dict = list(sent_analysis_data['Spanish Word'])
    negative_cols = ['negative','fear','anger','disgust','sadness']
    positive_cols = ['positive','joy','trust']

    negative_words = []
    for col in negative_cols:
        i = 0
        for val in sent_analysis_data[col]:
            if val == 1:
                negative_words.append(spanish_dict[i])
            i += 1

    positive_words = []
    for col in positive_cols:
        i = 0
        for val in sent_analysis_data[col]:
            if val == 1:
                positive_words.append(spanish_dict[i])
            i += 1

    return set(positive_words),set(negative_words)

def sent_analysis(words,positive_words,negative_words):

    intersection_neg = list(negative_words & set(words))
    intersection_pos = list(positive_words & set(words))

    return len(intersection_pos),len(intersection_neg)

# returns a list of words that occur exactly 'num' times or None if no coincidence
def num_occurences(lista,num):
    aux_dict = {}
    for item in lista:
        if item in aux_dict.keys():
            aux_dict[item] += 1
        else:
            aux_dict[item] = 1

    try:
        idx = list(aux_dict.values()).index(num)
        words = list(aux_dict.keys())[idx]
        return len(words)
    except:
        return 0
    
def get_region(provincia):
    
    df = pd.read_excel('../Tabla Provincias.xls')
    ccaa = list(df.loc[df['Nombre Provincia'].str.contains(provincia.upper())]['Nombre CCAA'])
    if len(ccaa) == 0:
        return provincia
    
    return ccaa[0]

def get_edad(fn):
    # fn del policia
    if '/' not in fn:
        year = fn[:4]
        month = fn[4:6]
        day = fn[6:]
    else:
        aux = fn.split('/')
        day = aux[0]
        month = aux[1]
        year = aux[-1]

    fecha = datetime.date(int(year),int(month),int(day))
    diff = relativedelta(datetime.date.today(),fecha).years

    if diff < 25:
        edad = '18-24'
    elif diff < 35:
        edad = '25-34'
    elif diff < 45:
        edad = '35-44'
    elif diff < 55:
        edad = '45-54'
    else:
        edad = '+55'
    return edad
        

def save_results(y_test,y_pred,average,tunning=False,params=None):

    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred,average=average)
    rec = recall_score(y_test,y_pred,average=average)
    f1 = f1_score(y_test,y_pred,average=average)

    if tunning:
        sent = f'\t* With tunning {params}:\n'
    else:
        sent = '\t* Without tunning:\n'

    with open('resultados_policia.txt','a') as f:
        f.write(sent)
        f.write(f'\t\t-> Accuracy: {acc}\n')
        f.write(f'\t\t-> Precision: {prec}\n')
        f.write(f'\t\t-> Recall: {rec}\n')
        f.write(f'\t\t-> F1-score: {f1}\n')


def load_data(embedding,feature_type):
    # load spanish stop words and remove accents (tweets dont have accents)
    stop_words_df = pd.read_csv('../../spanish-stop-words.txt',header=None)
    stop_words = list(stop_words_df[0])

    files = os.listdir('../Atestados')
    files.remove('.DS_Store')

    if embedding is not None:
        df = pd.DataFrame(columns=[str(i) for i in range(300)])
        print('loading embedding')
        embedding_vectors = load_embedding(embedding)
        print('embedding loaded')
    else:
        df = pd.DataFrame()

    if ('sf' in feature_type) or (feature_type == 'all'):
        print('loading sentiment analysis')
        positive_words,negative_words = load_sentiment_analysis()

    punctuation_list = list(string.punctuation)
    pos_tag = tagger()

    num_char = []
    num_capital = []
    num_punctuation = []
    num_sentence = []
    av_sentence_par = []
    av_words_par = []
    av_char_par = []
    variation = []
    num_det = []
    num_pre = []
    num_sing = []
    num_plural = []
    num_adv = []
    num_adj = []
    num_prop = []
    num_pronouns = []
    num_past = []
    num_future = []
    num_conj = []
    num_words = []
    num_pos_words = []
    num_neg_words = []
    num_unique = []
    num_twice = []
    av_length = []
    max_length = []
    num_numbers = []
    num_greater = []
    num_smaller = []
    num_stop = []
    num_locations = []
    num_telefono = []
    num_fechas = []
    num_personas = []
    list_genero = []
    list_edad = []
    list_region = []
    
    df_aux = pd.DataFrame()
    for file in files:
        list_sent_tweet = []

        with open(f'../Atestados/{file}') as f:
            _dict = json.load(f)
            # policia
            policia = _dict['cp']
            provincia = policia['provincia'].title()
            genero = policia['sexo']
            fn = policia['fn']
            region = get_region(provincia)
            edad = get_edad(fn)

            list_genero.append(genero)
            list_edad.append(edad)
            list_region.append(region)

            # denuncia
            text = _dict['denuncia']
            
            if ('sf' in feature_type) or (feature_type == 'all'):
                doc = text

                # before cleaning, count num of tags
                tags = re.findall('LOC|NUM|TIEM|PER',doc)
                counter = Counter(tags)
                for k in ['LOC','NUM','TIEM','PER']:
                    if k not in counter.keys():
                        counter[k] = 0
                        
                for k,v in counter.items():
                    if k == 'LOC':
                        num_locations.append(v)
                    elif k == 'NUM':
                        num_telefono.append(v)
                    elif k == 'TIEM':
                        num_fechas.append(v)
                    elif k == 'PER':
                        num_personas.append(v)

                # remove tags and '--'
                doc = quitar_tags(doc)
                doc = quitar_guiones(doc)
                doc = replace_multi_whitespaces(doc)

                # character based
                num_char.append(len(doc))
                num_capital.append(sum(1 for c in doc if c.isupper()))
                num_punctuation.append(sum(1 for c in doc if c in punctuation_list))

                # structural based
                doc = doc.lower()
                num_par = len(doc.split('\n'))

                aux = doc.replace('\n','')
                sentences = aux.split('.')
                num_sentence_user = len(sentences)
                num_sentence.append(num_sentence_user)
                av_sentence_par.append(num_sentence_user/num_par)
                
                words = [w for w in doc.split(' ') if len(w) > 0]
                av_words_par.append(len(words)/num_par)
                av_char_par.append(len(aux)/num_par)

                len_sentence_list = [len(sentence) for sentence in sentences]
                if len(len_sentence_list) > 1:
                    var = stat.variance(len_sentence_list)
                else:
                    var = 0
                variation.append(var)     

                # syntactical based
                # lowercase and remove punctuation marks
                doc = doc.translate(str.maketrans('','',string.punctuation))
                analysis = pos_tag.tag(doc.split(' '))

                det = 0
                pre = 0
                sing = 0
                plural = 0
                adv = 0
                adj = 0
                prop = 0
                pronouns = 0
                past = 0
                future = 0
                conj = 0

                for word,tag in analysis:
                    if tag != None:
                        if tag[0] == 'd':
                            det += 1
                        elif tag[0] == 'a':
                            adj += 1
                        elif tag[0] == 'c':
                            conj += 1
                        elif tag[0] == 'p':
                            pronouns += 1
                        elif tag[0] == 'n':
                            if tag[1] == 'p':
                                prop += 1
                            if tag[3] == 's':
                                sing += 1
                            elif tag[3] == 'p':
                                plural += 1
                        elif tag[0] == 'r':
                            adv += 1
                        elif (tag[0] == 'v' and tag[3] == 'f'):
                            future += 1
                        elif (tag[0] == 'v' and tag[3] == 's'):
                            past += 1
                        elif tag[0] == 's':
                            pre += 1

                
                num_det.append(det)
                num_pre.append(pre)
                num_sing.append(sing)
                num_plural.append(plural)
                num_adv.append(adv)
                num_adj.append(adj)
                num_prop.append(prop)
                num_pronouns.append(pronouns)
                num_past.append(past)
                num_future.append(future)
                num_conj.append(conj)

                # word based
                num_words.append(len(words))

                pos, neg = sent_analysis(words,positive_words,negative_words)

                num_pos_words.append(pos)
                num_neg_words.append(neg)

                # unique words
                num_unique.append(num_occurences(words,1))
                # twice occurrences
                num_twice.append(num_occurences(words,2))
                
                # max, av, >6, <3 length and num words with digits, count english words
                max_len = 0
                sum_length = 0
                digits = 0
                len_greater = 0
                len_smaller = 0
                for word in words:
                    sum_length += len(word)
                    if len(word) > max_len:
                        max_len = len(word)

                    if len(re.findall('\d',word)) > 0:
                        digits += 1

                    if len(word) > 6:
                        len_greater += 1
                    elif len(word) < 3:
                        len_smaller += 1
                
                av_length.append(sum_length/len(words))
                max_length.append(max_len)
                num_numbers.append(digits)
                num_greater.append(len_greater)
                num_smaller.append(len_smaller)

                # count stop-words 
                intersection_stop = list(set(stop_words) & set(words))
                num_stop.append(len(intersection_stop))
            

            if feature_type == 'emb' or feature_type == 'all':
                doc_emb = text
                doc_emb = quitar_guiones(doc_emb)
                sentences = doc_emb.split('\n')
                # embeddings
                for sent in sentences:
                    aux = []
                    # clean each sentence
                    cleaned = clean_text(sent,stop_words)
                    # compute vector for each word in the sentence using GloVe
                    for token in word_tokenize(cleaned,language='spanish',preserve_line=True):
                        vector = embedding_vectors.get(token)
                        if vector is not None:
                            aux.append(vector)

                    if len(aux) != 0:
                        vec_sent = np.asarray(aux).mean(0)
                        list_sent_tweet.append(vec_sent)
                # compute author's vector averaging tweets vectors
                vec_author = np.asarray(list_sent_tweet).mean(0)
                df.loc[len(df)] = vec_author

    gender_dict = {'M':0, 'H':1}

    age_dict = {}
    i = 0
    for age in set(list_edad):
        age_dict[age] = i
        i += 1

    region_dict = {}
    i = 0
    for region in set(list_region):
        region_dict[region] = i
        i += 1
    
    data = pd.DataFrame({'gender':list_genero,'age':list_edad,'region':list_region})

    df['gender'] = data['gender'].map(gender_dict)
    df['age'] = data['age'].map(age_dict)
    df['region'] = data['region'].map(region_dict)


    if ('sf' in feature_type) or (feature_type == 'all'):
        df_aux['characters'] = num_char
        df_aux['capital_letters'] = num_capital
        df_aux['punctuations'] = num_punctuation
        df_aux['num_sentence'] = num_sentence
        df_aux['av_sentence_par'] = av_sentence_par
        df_aux['av_words_par'] = av_words_par
        df_aux['av_char_par'] = av_char_par
        df_aux['variation'] = variation
        df_aux['num_det'] = num_det
        df_aux['num_pre'] = num_pre
        df_aux['num_sing'] = num_sing
        df_aux['num_plural'] = num_plural
        df_aux['num_adv'] = num_adv
        df_aux['num_adj'] = num_adj
        df_aux['num_prop'] = num_prop
        df_aux['num_pronouns'] = num_pronouns
        df_aux['num_past'] = num_past
        df_aux['num_future'] = num_future
        df_aux['num_conj'] = num_conj
        df_aux['num_words'] = num_words
        df_aux['num_pos_words'] = num_pos_words
        df_aux['num_neg_words'] = num_neg_words
        df_aux['num_unique'] = num_unique
        df_aux['num_twice'] = num_twice
        df_aux['av_length'] = av_length
        df_aux['max_length'] = max_length
        df_aux['num_numbers'] = num_numbers
        df_aux['num_greater'] = num_greater
        df_aux['num_smaller'] = num_smaller
        df_aux['num_stop'] = num_stop
        df_aux['num_locations'] = num_locations
        df_aux['num_fechas'] = num_fechas
        df_aux['num_personas'] = num_personas
        df_aux['num_telefono'] = num_telefono

    
    if feature_type != 'emb':
        # scale TF and SF
        scaler = MinMaxScaler(feature_range=(-1,1))
        df_scaled = pd.DataFrame(scaler.fit_transform(df_aux),columns=df_aux.columns)

        df = df.join(df_scaled)

    return df


def exec_model(model, df, target_col, features, feature_type, remove_outliers):
    if features is None:
        features = [str(i) for i in range(300)]
    else:
        if remove_outliers:
            # remove outliers
            for f in features:
                Q1 = np.percentile(df[f],25,method = 'midpoint')
                Q3 = np.percentile(df[f],75,method = 'midpoint')
                IQR = Q3 - Q1
                upper = Q3 + 1.5*IQR
                lower = Q1 - 1.5*IQR
                drop_index = np.where(df[f]>=upper) + np.where(df[f]<=lower)
                df.drop(drop_index[0], inplace=True)
                df.reset_index(drop=True,inplace=True)

        if feature_type == 'all':
            features += [str(i) for i in range(300)]
    
    target = df[target_col]
    col_drop = [f for f in df.columns if f not in features]
    df.drop(col_drop,axis=1,inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df, target, stratify=target, shuffle=True,test_size=0.2,random_state=109)

    if target_col == 'gender':
        average = 'binary'
        objective = 'binary:logistic'
    else:
        average = 'weighted'
        objective = 'multi:softmax'

    if model == 'svm':
        clf = svm.SVC() 
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        save_results(y_test,y_pred,average)
        print('results without tunning saved')

        # tunning
        params = {'C': 100, 'gamma': 1, 'kernel': 'rbf'}
        clf = svm.SVC(C=100,gamma=1,kernel='rbf')
        clf.fit(X_train, y_train)
    
    elif model == 'rf':
        clf = RandomForestClassifier()
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        save_results(y_test,y_pred,average)
        print('results without tunning saved')

        params = {'criterion': 'gini', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
        clf = RandomForestClassifier(criterion='gini',max_leaf_nodes=None,min_samples_leaf=1,min_samples_split=2,n_estimators=500)
        clf.fit(X_train, y_train)
        
    elif model == 'xgboost':
        clf = xgb.XGBRFClassifier(objective=objective)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        save_results(y_test,y_pred,average)
        print('results without tunning saved')

        params = {'learning_rate': 0.01, 'n_estimators': 100, 'reg_lambda': 1}
        clf = xgb.XGBRFClassifier(learning_rate=0.01,n_estimators=100,reg_lambda=1,objective=objective)
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    save_results(y_test,y_pred,average,tunning=True,params=params)

def main():
    
    if len(sys.argv) < 6:
        print('Please enter model, target, embedding. i.e "python word_embeddings.py "svm" "gender" "glove" "remove" "tf"')
        return
    
    model = sys.argv[1]
    target = sys.argv[2]

    if sys.argv[3] == 'glove':
        embedding = 'glove-sbwc.i25.vec'
    elif sys.argv[3] == 'fasttext':
        embedding = 'embeddings-l-model.vec'
    elif sys.argv[3] == 'none':
        embedding = None
    else:
        print('Please enter a valid embedding: glove, fasttext, none')
        return
    
    remove_outliers = False
    features = None

    if sys.argv[4] == 'remove':
        remove_outliers = True
    
    feature_type = sys.argv[5]
    if feature_type == 'sf' or feature_type == 'all':
        features = {'gender': ['characters','capital_letters','num_sentence','av_sentence_par','num_det','num_adv','num_pronouns','num_words','av_length','num_smaller','num_stop'],
                    'age': ['characters','capital_letters','punctuations','num_sentence','av_sentence_par','num_pre','num_sing','num_plural','num_adj','num_pronouns','num_conj','num_words','num_unique','num_twice','av_length','num_numbers','num_greater','num_smaller','num_fechas'],
                    'region': ['characters','capital_letters','av_words_par','av_char_par','num_det','num_pronouns','num_words']
                    }
    elif feature_type == 'emb':
        features = None
    else: 
        print('Enter a valid feature: emb, sf, all')


    print('loading data')
    df = load_data(embedding,feature_type)
    print('data loaded')
    if features != None:
        exec_model(model,df,target,features.get(target),feature_type,remove_outliers)
    else:
        exec_model(model,df,target,features,feature_type,remove_outliers)

if __name__ == '__main__':
    main()
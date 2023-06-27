from sklearn.model_selection import train_test_split
import pandas as pd
import fasttext
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import sys
import json
import datetime
from dateutil.relativedelta import relativedelta
import re

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

def quitar_guiones(text):
    aux = text.split('--')
    aux2 = []
    for w in aux:
        if len(w)>0:
            text = replace_multi_whitespaces(w)
            aux2.append(text)
    return '\n'.join(aux2)

def replace_multi_whitespaces(line):
    return ' '.join(line.split())

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
    text = quitar_guiones(text)
    text = quitar_tags(text)
    text = remove_stopwords(text,stop_words)
    text = replace_multi_whitespaces(text)
    return text        

def load_df():
    files = os.listdir('../Atestados')
    files.remove('.DS_Store')

    stop_words_df = pd.read_csv('../../spanish-stop-words.txt',header=None)
    stop_words = list(stop_words_df[0])

    df = pd.DataFrame()

    list_genero = []
    list_edad = []
    list_region = []
    denuncia = []

    for file in files:
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
            text = clean_text(_dict['denuncia'],stop_words)
            denuncia.append(text)

    df['gender'] = list_genero
    df['age'] = list_edad
    df['region'] = list_region
    df['text'] = denuncia

    return df

def main():

    if len(sys.argv) < 2:
        print('Please specify the target variable. Options: gender, age, region')
        return
    
    target = str(sys.argv[1])

    data = load_df()
    data_train,data_test = train_test_split(data,test_size=0.3,shuffle=True,stratify=data[target])

    unique_values = data[target].unique()
    unique_values = [u.replace(' ','') for u in unique_values]
    i = 0
    labels = {}
    for l in unique_values:
        labels[l] = f'__label__{l} '

    file_train = 'data_train.txt'
    with open(file_train,'x') as f:
        for text,label in zip(data_train['text'],data_train[target]):
            text += f' {labels.get(label)}\n'
            f.write(text)

    file_test = 'data_test.txt'
    with open(file_test,'x') as f:
        for text in data_test['text']:
            f.write(text+'\n')

    i = 0
    target_dict = {}
    for t in data[target].unique():
        target_dict[t] = i
        i+=1

    y_test = list(data_test[target].map(target_dict))

    model = fasttext.train_supervised(input=file_train,
                                      lr=0.025,
                                      epoch=20,
                                      dim=300,
                                      minn=2,
                                      maxn=7,
                                      wordNgrams=3,
                                      pretrainedVectors='../../embeddings/embeddings-l-model.vec')

    y_pred= []

    with open(file_test,'r') as f:
        lines = f.read().splitlines()

    for line in lines:
        label,prob = model.predict(line)
        val = target_dict.get(label[0].replace('__label__',''))
        y_pred.append(val)
    
    if target == 'gender':
        av = 'binary'
    else:
        av = 'weighted'

    precision = precision_score(y_test, y_pred,average=av)
    recall = recall_score(y_test, y_pred,average=av)
    f1 = f1_score(y_test, y_pred,average=av)
    accuracy = accuracy_score(y_test, y_pred)

    with open('results.txt','a') as f:
        f.write(f'\t-> Accuracy: {accuracy}\n')
        f.write(f'\t-> Precision: {precision}\n')
        f.write(f'\t-> Recall: {recall}\n')
        f.write(f'\t-> F1-score: {f1}\n')

    os.remove(file_train)
    os.remove(file_test)



if __name__ == '__main__':
    main()
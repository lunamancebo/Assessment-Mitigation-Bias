from corpus_processing import main as preprocessing
from corpus_processing import get_labels
from sklearn.model_selection import train_test_split
import pandas as pd
import fasttext
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import sys

def main():

    if len(sys.argv) < 2:
        print('Please specify the target variable. Options: gender, age, region')
        return
    
    target = str(sys.argv[1])

    data = pd.read_excel('../cleaned_users.xlsx')
    data_train,data_test = train_test_split(data,test_size=0.3,shuffle=True,stratify=data[target])

    unique_values = data[target].unique()
    unique_values = [u.replace(' ','') for u in unique_values]
    i = 0
    labels = []
    for l in unique_values:
        labels.append((f'__label__{l}',i))
        i+=1

    file_train = preprocessing(data_train,'data_train',target)
    file_test = preprocessing(data_test,'data_test',target,label=False)
    y_test = get_labels(data_test,target,labels)

    model = fasttext.train_supervised(input=file_train,
                                      lr=0.025,
                                      epoch=20,
                                      dim=300,
                                      minn=2,
                                      maxn=7,
                                      wordNgrams=3,
                                      pretrainedVectors='embeddings-l-model.vec')

    y_pred= []

    with open(file_test,'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        aux = False
        label,prob = model.predict(line)
        for l, val in labels:
            if label[0] == l:
                aux = True
                y_pred.append(val)

        if not aux:
            print(label[0])

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
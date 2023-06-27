'''
Concatenate tweets of the same author in a single document,
resulting in one document per author.
Consider only the users with more than 10 tweets.
'''

import json
import pandas as pd

data = pd.read_excel('twitter_users.xlsx')
data.drop_duplicates(subset='username',inplace=True)
username_list = list(data['username'])
gender_init = list(data['gender'])
age_init = list(data['age'])
region_init = list(data['region'])

users_list = []
gender_list = []
age_list = []
region_list = []

i = 0
for username,gender,age,region in zip(username_list,gender_init,age_init,region_init):
    text_doc = ''

    try:
        user_json = pd.read_json(f'Tweets/{username}.json',typ=dict)

        if len(user_json['data']) >= 10:
            for tweet in user_json['data']: 
                text = tweet['text']
                if text[-1] != '.':
                    text += '.\n'
                else:
                    text += '\n'
                text_doc += text

            with open(f'Documents/{username}.txt','x') as f:
                f.write(text_doc)

            users_list.append(username)
            gender_list.append(gender)
            age_list.append(age)
            region_list.append(region)
            
    except:
        continue

    i += 1


data_dict = {'username':users_list,
             'gender':gender_list,
             'age':age_list,
             'region':region_list}

new_data = pd.DataFrame(data_dict)
new_data.to_excel('cleaned_users.xlsx',index=False)
            

    
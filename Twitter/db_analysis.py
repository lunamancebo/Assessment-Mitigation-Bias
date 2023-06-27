import pandas as pd

tweet_df = pd.read_excel('twitter_users.xlsx')

gender_dict = {}
age_dict = {}
region_dict = {}

for username,gender,age,region in zip(tweet_df['username'],tweet_df['gender'],tweet_df['age'],tweet_df['region']):

    if gender in gender_dict.keys():
        gender_dict[gender].append(username)
    else:
        gender_dict[gender] = [username]
    
    if age in age_dict.keys():
        age_dict[age].append(username)
    else:
        age_dict[age] = [username]
    
    if region in region_dict.keys():
        region_dict[region].append(username)
    else:
        region_dict[region] = [username]

total = len(tweet_df)
#freq gender
print('\n------- FREQUENCY GENDER -------\n')
for key in gender_dict.keys():
    freq = (len(gender_dict[key])/total)*100
    print(f'{key}: {freq} %')

#freq age
print('\n-------- FREQUENCY AGE ---------\n')
for key in age_dict.keys():
    freq = (len(age_dict[key])/total)*100
    print(f'{key}: {freq} %')

#freq region
print('\n------- FREQUENCY REGION -------\n')
for key in region_dict.keys():
    freq = (len(region_dict[key])/total)*100
    print(f'{key}: {freq} %')
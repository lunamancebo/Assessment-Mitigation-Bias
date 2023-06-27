from tweepy import Client, TwitterServerError, HTTPException
import os
import json
from datetime import datetime
import pandas as pd
from time import sleep

def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ü","u"),
        ("ñ","n"),
        ("ç","c"),
        ("\u2026","..."),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def write_data(retrieved_tweets, username):
    print('Writing json...')
    retrieved_data = {'data':retrieved_tweets}
    with open(f'Tweets/{username}.json', "w") as f:
        json.dump(retrieved_data, f, indent=4)

# Read excel and remove duplicate values
raw_data = pd.read_excel('twitter_users.xlsx')
data = raw_data.drop_duplicates(subset='username')
username_list = list(data['username'])[657:]


# API keyws that yous saved earlier
bearer_token = "AAAAAAAAAAAAAAAAAAAAAEusiQEAAAAAozeeCWU4NgsfFjM69%2F6mL01qCjg%3DjnZfI6F3IvCQ0hvCefKNPw95VrTu39d99NLgOIkTZaMyQ5tkcF"
consumer_key = "zG4Eg3z7sKQ7XfY5XR166LAfW"
consumer_secret = "mAyzKN7v31d4IeOd17kxcPIpIcuo1LZSpypSWFwlaHg7gJzYVA"
access_token = "1582469263356485641-JVemoSu2Nax9T4tJGjQq4oKL6Ua9kd"
access_token_secret = "SclOsKtAFyaPEBvL93teSn0JE5rOiRtKGLzMdTyFKSGkZ"

# Create a client
client = Client(bearer_token=bearer_token,return_type=dict,wait_on_rate_limit=True)
start_time = datetime(2018,1,1)
for username in username_list:
    # Sleep 1sec to avoid rate limit
    sleep(1.1)
    num_tweets = 0
    retrieved_tweets = []
    query = f'from:{username} lang:es -is:retweet'

    try:
        response = client.search_all_tweets(query=query, 
                                            max_results=100,
                                            start_time=start_time,
                                            tweet_fields = ['author_id','text','created_at'],
                                            expansions='author_id')

    except TwitterServerError:
        write_data(retrieved_tweets,username)

    except HTTPException as err:
        print(f'Username: {username}')
        print('ERRORS:')
        print('Errors: ' + err.api_errors)
        print('Codes: ' + err.api_codes)
        print('Messages: ' + err.api_messages)

    if response['meta']['result_count'] > 0:
        for tweet in response['data']:
            d = {'tweet_id':tweet['id'],
                'text':normalize(tweet['text']),
                'created_at':tweet['created_at'],
                'user_id':tweet['author_id'],
                'username':username}
            
            retrieved_tweets.append(d)

    num_tweets += response['meta']['result_count']
    print(num_tweets)

    if len(retrieved_tweets) > 0:
        write_data(retrieved_tweets,username)

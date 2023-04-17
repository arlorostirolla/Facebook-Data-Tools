import pandas as pd
import re, nltk, json, os, datetime
import matplotlib.pyplot as plt
import pickle
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.interpolate import interp1d

def parse(myName):
    if not os.path.exists("./per_person/"):
        os.mkdir("./per_person/")
    OTHER = ''
    count = 0
    print("Parsing conversations")
    for path, dirc, files in os.walk('./data/'):
        print(str(count) + " Conversations parsed \r", end="")
        count+=1
        for name in files:
            if name.endswith('.json'):
                with open(path+'/'+name, "r") as file:
                    data = json.load(file)

                    if isinstance(data, list) or not 'participants' in data.keys():
                        continue

                    participants = data['participants']
                    OTHER = [i for i in participants if i != myName][0]['name']

                    if len(participants) <= 2:
                        messages = data['messages']
                        df = pd.DataFrame(messages)
                        if not 'content' in df.keys():
                            continue
                        if OTHER + '.csv' in os.listdir("./per_person"):
                            previous = pd.read_csv("./per_person/" + OTHER + '.csv')
                            df = pd.concat([previous, df])
                        
                        df = df.sort_values(by ="timestamp_ms")
                        try:
                            df.to_csv("./per_person/" + OTHER + '.csv')
                        except:
                            print("Error with " + OTHER)

def clean():
    print("Cleaning conversations")
    data['datetime'] = pd.to_datetime(data.timestamp_ms, unit='ms')
    for path, dirc, files in os.walk('./per_person/'):
        for name in files:
            if name.endswith('.csv'):
                df = pd.read_csv(path+'/'+name)
                df = df.dropna(subset=['content'])
                df['content'] = df['content'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
                df['content'] = df['content'].apply(lambda x: x.lower())
                df['content'] = df['content'].apply(lambda x: re.sub(r'\s+', ' ', x))
                df.to_csv(path+'/'+name)

def plotSentimentOverTime(data, average=100):
    data['datetime'] = pd.to_datetime(data.timestamp_ms, unit='ms')
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    def calculate_sentiment(row):
        try:
            row['content'] = str(row['content'].encode("utf-8"))
        except AttributeError:
            return
        
        row['compound'] = sia.polarity_scores(row['content'])['compound']
        return row
    
    data = data.apply(calculate_sentiment, axis=1)
     
    plt.figure(figsize=(20,20))
    data = data.sort_values(by='timestamp_ms', ascending=True)
    plt.xticks(rotation=45)
    data[ 'rolling_avg'] = data.compound.rolling(average).mean()
    
    plt.plot(data['datetime'], data['rolling_avg'])
    # the following line is to smooth out the plot

    plt.show()

def analyse_time_to_reply(data, person, average=100):
    data['datetime'] = pd.to_datetime(data.timestamp_ms, unit='ms')
    my_time_to_reply = []
    their_time_to_reply = []
    for i in range(1, data.shape[0]):
        previous_name = data.iloc[i-1]['sender_name']
        current_name = data.iloc[i]['sender_name']
        if previous_name == current_name:
            continue
        else:
            if previous_name == person:
                my_time_to_reply.append([data.iloc[i]['datetime'], (data.iloc[i]['timestamp_ms'] - data.iloc[i-1]['timestamp_ms'])/3.6e+6])
            else:
                their_time_to_reply.append([data.iloc[i]['datetime'], (data.iloc[i]['timestamp_ms'] - data.iloc[i-1]['timestamp_ms'])/3.6e+6
])
    
    out = pd.DataFrame(their_time_to_reply, columns=['datetime', 'time_to_reply'])
    out['rolling_avg'] = out['time_to_reply'].rolling(average).mean()
    print(out.head())
    plt.plot(out['datetime'], out['rolling_avg'])
    #plt.plot([x[0] for x in their_time_to_reply], [x[1] for x in their_time_to_reply])
    plt.show()
                



if __name__ == "__main__":
    #parse("Arlo Rostirolla")
    #clean()
    data = pd.read_csv("./per_person/_______.csv")
    #plotSentimentOverTime(data, average=300)
    analyse_time_to_reply(data, "__________")

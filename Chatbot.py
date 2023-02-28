import pandas as pd
import re, nltk, json, os, datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import language_tool_python as ltp
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
import pickle
import numpy as np

NAME = 'Arlo Rostirolla'

class FBParser():
    def __init__(self, datapath):
        self.datapath = datapath
        self.data = self.parse()
        self.vocabulary = set()
        self.occurrances = dict()

    def parse(self):
        global name
        if os.path.exists("./uncleaned_dataframe.pkl"):
            print("Loading dataframe")
            return pd.read_pickle("./uncleaned_dataframe.pkl")
        JSON = ('.json')
        OTHER = ''
        dfs = []
        count = 0
        print("Parsing conversations")
        for path, dirc, files in os.walk(self.datapath):
            print(str(count) + " Conversations parsed \r", end="")
            count+=1
            for name in files:
                promptAnswer = []
                if name.endswith(JSON):
                    with open(path+'/'+name, "r") as file:
                        data = json.load(file)
                        participants = data['participants']
                        OTHER = [i for i in participants if i != NAME][0]['name']
                        if len(participants) <= 2:
                            messages = data['messages']
                            df = pd.DataFrame(messages)
                            df = df.drop(["sender_id_INTERNAL", "type", "is_unsent", "is_taken_down", "bumped_message_metadata"], axis=1)
                            if not 'content' in df.keys():
                                continue
                            df = df.sort_values(by ="timestamp_ms")
                            prompts = ["", ""]
                            meString = ""
                            otherString = ""
                            index = 0
                            for i in range(df.shape[0]):
                                if df.iloc[index]['sender_name'] == NAME:
                                    continue
                                else:
                                    
                                    index = i
                                    break
                            previous_name = df.iloc[index]['sender_name']

                            for i in range(index, df.shape[0]):
                                current_name = df.iloc[i]['sender_name']
                                current_timestamp = df.iloc[i]['timestamp_ms']
                                if current_name == NAME:
                                    meString += ' '+str(df.iloc[i]["content"])+'.'
                                if current_name != NAME:
                                    otherString += ' '+str(df.iloc[i]["content"])+'.'
                                if previous_name != NAME and current_name == NAME:
                                    prompts[0] = otherString
                                    otherString = ""
                                if previous_name == NAME and current_name != NAME:
                                    prompts[1] = meString
                                    meString = ""
                                if prompts[0] != "" and prompts[1] != "":
                                    promptAnswer.append({'username': OTHER, 'prompt': prompts[0], 'answer': prompts[1], 'timestamp': df.iloc[i]["timestamp_ms"]})
                                    prompts = ["", ""]
                                previous_name = current_name
                                
                if len(promptAnswer) > 1:
                    promptAnswerDF = pd.DataFrame(promptAnswer)
                    promptAnswerDF.sort_values(by ="timestamp", inplace=True)
                    dfs.append(promptAnswerDF)
                
        output = pd.concat(dfs)
        print(output.columns)
        print(output.shape[0])
        print(output.describe())
        output.to_pickle("./uncleaned_dataframe.pkl") 
        return output

    def clean_data(self):
        phrases = ["You can now message and call each other and see info like Active Status and when you've read messages",
                    "The video chat ended.", "missed your video chat", "missed your call", "You are now connected on Messenger."]
        
        
        print("Cleaning data")
        self.count = 0
        def clean(row):
            self.count += 1
            print(str(self.count) + " utterances cleaned \r", end="")

            text = row['answer']
            # remove facebook automessages
            for i in phrases:
                text = text.replace(i, "")
            # remove URLs, mentions and non english characters
            text = re.sub("@[A-Za-z0-9]+", " ", text)
            text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", " ", text)
            text = re.sub("[^A-Za-z0-9]", " ", text)
            row['answer'] = text
            return row  

        if os.path.exists("./cleaned_dataframe.pkl"):
            print("Reloading dataframe")
            self.data = pd.read_pickle("./cleaned_dataframe.pkl")
            return

        before = self.data.shape[0]
        self.data = self.data.apply(clean, axis=1)
        after = self.data.shape[0]
        self.data.to_pickle("./cleaned_dataframe.pkl") 
        print("Finished cleaning")

    def dropBasedOnSentiment(self, cutoff):
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        self.count = 0
        def calculate_sentiment(row):
            self.count += 1
            print(str(self.count) + "sentiments calculated \r", end="")
            row['compound'] = sia.polarity_scores(row['answer'])['compound']
            return row

        self.data = self.data.apply(calculate_sentiment, axis=1)
        before = self.data.shape[0]
        self.data = self.data[self.data.compound >= cutoff]
        after = self.data.shape[0]
        print(f"{before-after} rows removed based on sentiment")
        self.data.to_pickle("./cleaned_dataframe.pkl") 

    def dropBasedOnTimestampRange(self, from_timestamp, to_timestamp):
        def drop_row(row):
            if row.iloc[3] >= from_timestamp or row.iloc[3] <= to_timestamp:
                return None
            else:
                return row

        before = self.data.shape[0]
        self.data = self.data[~((self.data['timestamp'] >= from_timestamp) & (self.data['timestamp'] <= to_timestamp))]
        after = self.data.shape[0]
        print(f"{before-after} rows removed based on date range")
        self.data.to_pickle("./cleaned_dataframe.pkl") 

    def dropBasedOnUsername(self, username):
        before = self.data.shape[0]
        self.data = self.data[self.data['username'] != username]
        after = self.data.shape[0]
        print(f"{before-after} rows removed based on username")
        self.data.to_pickle("./cleaned_dataframe.pkl") 

    def dropBasedOnGrammar(self, matches_cutoff):
        toGo = []
        my_tool = ltp.LanguageTool('en-AU')  
        self.count = 0
        def clean(row):
            self.count += 1
            print(str(self.count) + " utterances checked for grammar \r", end="")
            my_matches = my_tool.check(row['answer'])  
            if len(my_matches) > matches_cutoff:
                return None
            else: 
                return row
        
        before = self.data.shape[0]
        self.data = self.data.apply(clean, axis=1)
        after = self.data.shape[0]
        print(f"{before-after} rows removed based on grammar")
        self.data.to_pickle("./cleaned_dataframe.pkl") 
    
    def dropBasedOnWords(self, words_list):
        before = self.data.shape[0]
        self.count = 0
        def clean(row):
            self.count += 1
            for word in words_list:
                if any([i in row['answer'].split() for i in words_list]):
                    row['contains'] = True
                    return row
            print(str(self.count) + " rows parsed \r", end="")
            row['contains'] = False
            return row
    
        self.data = self.data.apply(clean, axis=1)
        self.data = self.data[self.data['contains'] == False]
        after = self.data.shape[0]
        print(f"{before-after} rows removed based on contained words")
        self.data.to_pickle("./cleaned_dataframe.pkl") 

    def saveUtterances(self):
        if os.path.exists("./cleaned_dataframe.pkl"):
            print("Reloading dataframe")
            self.data = pd.read_pickle("./cleaned_dataframe.pkl")
        
        output = pd.DataFrame(columns=['file_id', 'context', 'response'])
        print("Saving cleaned data to JSON for transformer model")
        self.count = 0
        for username in self.data.username.unique():
            self.count+=1
            mask = self.data[self.data['username'] == username].sort_values(by='timestamp')
            for index, row in mask.iterrows():
                output = output.append({'file_id': f'{username}.json', 'context': row['prompt'], 'response': row['answer']}, ignore_index=True)
            print("\r"+f"{self.count} user conversations converted", end="")
        output.to_json('./utterances.json', orient='records', lines=True)

class EDA():
    def __init__(self, data):
        self.data = data
        
    def TopicMining(self):
        from uctopic import UCTopic, UCTopicTokenizer, UCTopicTool
        import nltk
        nltk.download('stopwords')
        if os.path.exists("./cleaned_dataframe.pkl"):
            print("Reloading dataframe")
            self.data = pd.read_pickle("./cleaned_dataframe.pkl")
        
        model = UCTopic.from_pretrained('JiachengLi/uctopic-base')
        tokenizer = UCTopicTokenizer.from_pretrained('JiachengLi/uctopic-base')
        topic_tool = UCTopicTool('JiachengLi/uctopic-base', device='cuda:0')
        inputs = [i for i in self.data['answer']] + [i for i in self.data['prompt']]
        output_topics, topics_dict = topic_tool.topic_mining([i for i in inputs],\
                                                   n_clusters=[2, 5])
        with open('./output_topics.pkl', 'wb') as file:
            pickle.dump(output_topics, file)
        with open('./output_topics.pkl', 'wb') as file:
            pickle.dump(output_topics, file)
        topic_tool.save('./')
                                                   
    def concordances(self, word, num):
        all_messages = " ".join([i for i in self.data['prompt']] + [i for i in self.data['answer']])
        corpus = nltk.Text(nltk.wordpunct_tokenize(all_messages))
        concordances = corpus.concordance(word, lines=num)

    def plotSentimentHistogram(self, users=False):
        global name
        if users == False:
            plt.hist(self.data[self.data['username'] == name]['compound'], bins=100)
        else:
            plt.hist(self.data['compound'], bins=100)
        plt.show()

    def plotSentimentOverTime(self, average=20):
        plt.figure(figsize=(20,20))
        self.data = self.data.sort_values(by='timestamp', ascending=True)
        plt.xticks(rotation=45)
        if average > 0:
            averages = ["" for i in range(self.data['compound'].shape[0])]
            for i, v in enumerate(self.data['compound']):
                if i < average:
                    averages[i] = np.mean(self.data['compound'].iloc[[i-j for j in range(i)]])
            else:
                averages[i] = np.mean(self.data['compound'].iloc[[i-j for j in range(average)]])
        plt.plot(self.data['datetime'], averages)
        
if __name__ == '__main__':
    fbparse = FBParser('./conversations')
    print("Clean Data? (Y/N)")
    print("This will remove URLs, mentions, non english characters and fix spelling mistakes")
    if input().lower() == 'y':
        fbparse.clean_data()

    print('Drop based on timestamp range? (Y/N)')
    answer = input()
    if answer == 'Y':
        max_timestamp = fbparse.data['timestamp'].max()
        min_timestamp = fbparse.data['timestamp'].min()

        fbparse.data['datetime'] = pd.to_datetime(fbparse.data.timestamp, unit='ms')

        max_datetime = fbparse.data['datetime'].max()
        min_datetime = fbparse.data['datetime'].min()

        print(f'The minimum date that can be entered is {min_datetime} and the maximum date is {max_datetime}')
        print('Enter from date (YYYY MM DD)')
        from_date = input()
        print('Enter to date (YYYY MM DD')
        to_date = input()
        from_date = datetime.datetime.strptime(from_date, '%Y %m %d')
        to_date = datetime.datetime.strptime(to_date, '%Y %m %d')
        from_timestamp = from_date.timestamp() * 1000
        to_timestamp = to_date.timestamp() * 1000
        while from_timestamp < min_timestamp or to_timestamp > max_timestamp:
            print("Date entered is outside of the allowed range")
            print('Enter from date (YYYY MM DD)')
            from_date = input()
            print('Enter to date (YYYY MM DD')
            to_date = input()
            from_date = datetime.datetime.strptime(from_date, '%Y %m %d')
            to_date = datetime.datetime.strptime(to_date, '%Y %m %d')
            from_timestamp = from_date.timestamp() * 1000
            to_timestamp = to_date.timestamp() * 1000
        fbparse.dropBasedOnTimestampRange(from_timestamp, to_timestamp)

    print('Drop messages based on username? (Y/N)')
    answer = input()
    if answer.lower() == 'y':
        print('Enter username')
        username = input()
        print(fbparse.data['username'])
        while not fbparse.data['username'].isin([username]).any():
            print("Username not found")
            print('Enter username')
            username = input()
        fbparse.dropBasedOnUsername(username)
    
    print('Drop messages based on grammar? (Y/N)')
    answer = input()
    if answer.lower() == 'y':
        print('Enter number of grammar mistakes allowed (Integer)')
        num_matches = ""
        while not isinstance(num_matches, int):
            try:
                num_matches = int(input())
            except ValueError:
                print("please enter an integer")
        fbparse.dropBasedOnGrammar(num_matches)

    print('Drop based on sentiment classification? (Y/N')
    answer = input()
    if answer.lower() == 'y':
        print('Enter sentiment cutoff value. Should be between:\n   \
               -1.0 (most negative sentiment) to 1.0 (most positive)\n \
                Choose 0.0 to only allow positive sentiment data')
        cutoff = ""
        while not isinstance(cutoff, float):
            try:
                cutoff = float(input())
            except ValueError:
                print('Invalid input. Please enter a float value')
        fbparse.dropBasedOnSentiment(cutoff)

    print('Drop based on words? (Y/N)')
    answer = input()
    if answer.lower() == 'y':
        print('Enter words to drop (separated by commas)')
        words_list = input().split(',')
        fbparse.dropBasedOnWords(words_list)
    fbparse.saveUtterances()
   
                    

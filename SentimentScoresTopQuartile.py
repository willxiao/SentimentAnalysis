import sqlite3
import pandas as pd
from nltk.corpus import sentiwordnet as swn
import numpy as np

#Parts of code sampled from stackoverflow
#Calculates number of comments in the top quartile of every sentiment


sql_conn = sqlite3.connect('E:\Python\SentimentAnalysis\input\database.sqlite')

#We only want scores of comments that are long enough and have scores greater than 1
df = pd.read_sql("SELECT score, body FROM May2015 WHERE score > 1 AND LENGTH(body) > 5 AND LENGTH(body) < 250 LIMIT 10000", sql_conn)

keywords = ['Positive', 'Negative', 'Objective']

content_summary = pd.DataFrame()

def get_scores(x):
    return list(swn.senti_synsets(x))

def get_positive_score(sentiments):
    if len(sentiments) > 0:
        return sentiments[0].pos_score()
    return 0

def get_negative_score(sentiments):
    if len(sentiments) > 0:
        return sentiments[0].neg_score()
    return 0

def get_objective_score(sentiments):
    if len(sentiments) > 0:
        return sentiments[0].obj_score()
    return 0

pos_content = []
neg_content = []
obj_content = []

# get the average score for all words in the comments
for string in df['body'].values:
    strings = string.split(" ")
    string_scores = list(map(lambda x: get_scores(x), strings))
    pos_scores = list(map(lambda x: get_positive_score(x), string_scores))
    neg_scores = list(map(lambda x: get_negative_score(x), string_scores))
    obj_scores = list(map(lambda x: get_objective_score(x), string_scores))

    pos_content.append(np.mean(pos_scores))
    neg_content.append(np.mean(neg_scores))
    obj_content.append(np.mean(obj_scores))

df['Positive'] = pos_content
df['Negative'] = neg_content
df['Objective'] = obj_content


# we only want the  top quartile
pos_mean = np.mean(df['Positive'].values)
pos_content = df[df.Positive.apply(lambda x: x > pos_mean * 2.5)]
content_summary['Positive'] = pos_content.describe().score

neg_mean = np.mean(df['Negative'].values)
neg_content = df[df.Negative.apply(lambda x: x > neg_mean * 2.5)]
content_summary['Negative'] = neg_content.describe().score

obj_mean = np.mean(df['Objective'].values)
obj_content = df[df.Objective.apply(lambda x: x > obj_mean * 2.5)]
content_summary['Objective'] = obj_content.describe().score

print(df)


import sqlite3
import pandas as pd
from nltk.corpus import sentiwordnet as swn
import numpy as np

#Parts of code sampled from stackoverflow
#Calculates average scores of reddit comments grouped by sentiments


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

print(df)


import pandas as pd
import csv

filename_sanders = "./data/twitter_corpus-master/full-corpus.csv"
filename_gop = "./data/first-gop-debate-twitter-sentiment/Sentiment.csv"
filename_airline = "./data/twitter-airline-sentiment/Tweets.csv"
new_filename = "./data/tweets_all.csv"

tweets = pd.read_csv(filename_sanders)

print("Shape of tweets in Sanders dataset: " + str(tweets.shape))

processed_tweets_sanders = tweets.loc[tweets.Sentiment.isin(['positive', 'negative', 'neutral']), ['Sentiment', 'TweetText']]

print("Shape of processed tweets in Sanders dataset: " + str(processed_tweets_sanders.shape))
print("Positive tweets: " + str(len(processed_tweets_sanders.loc[processed_tweets_sanders.Sentiment == 'positive'])))
print("Negative tweets: " + str(len(processed_tweets_sanders.loc[processed_tweets_sanders.Sentiment == 'negative'])))
print("Neutral tweets: " + str(len(processed_tweets_sanders.loc[processed_tweets_sanders.Sentiment == 'neutral'])))

tweets = pd.read_csv(filename_gop)

print("Shape of tweets in GOP dataset: " + str(tweets.shape))

processed_tweets_gop = tweets.loc[tweets.sentiment.isin(['Positive', 'Negative', 'Neutral']), ['sentiment', 'text']]
processed_tweets_gop = processed_tweets_gop.rename(columns={'sentiment': 'Sentiment', 'text': 'TweetText'})
processed_tweets_gop['Sentiment'] = processed_tweets_gop['Sentiment'].replace({'Positive': 'positive', 'Negative': 'negative', 'Neutral': 'neutral'})

print("Shape of processed tweets in GOP dataset: " + str(processed_tweets_gop.shape))
print("Positive tweets: " + str(len(processed_tweets_gop.loc[processed_tweets_gop.Sentiment == 'positive'])))
print("Negative tweets: " + str(len(processed_tweets_gop.loc[processed_tweets_gop.Sentiment == 'negative'])))
print("Neutral tweets: " + str(len(processed_tweets_gop.loc[processed_tweets_gop.Sentiment == 'neutral'])))

tweets = pd.read_csv(filename_airline)

print("Shape of tweets in Airline dataset: " + str(tweets.shape))

processed_tweets_airline = tweets.loc[tweets.airline_sentiment.isin(['positive', 'negative', 'neutral']), ['airline_sentiment', 'text']]
processed_tweets_airline = processed_tweets_airline.rename(columns={'airline_sentiment': 'Sentiment', 'text': 'TweetText'})

print("Shape of processed tweets in Airline dataset: " + str(processed_tweets_airline.shape))
print("Positive tweets: " + str(len(processed_tweets_airline.loc[processed_tweets_airline.Sentiment == 'positive'])))
print("Negative tweets: " + str(len(processed_tweets_airline.loc[processed_tweets_airline.Sentiment == 'negative'])))
print("Neutral tweets: " + str(len(processed_tweets_airline.loc[processed_tweets_airline.Sentiment == 'neutral'])))

tweets_all = pd.concat([processed_tweets_sanders, processed_tweets_gop, processed_tweets_airline], ignore_index=True)
print("Shape of all tweets: " + str(tweets_all.shape))
print("Positive tweets: " + str(len(tweets_all.loc[tweets_all.Sentiment == 'positive'])))
print("Negative tweets: " + str(len(tweets_all.loc[tweets_all.Sentiment == 'negative'])))
print("Neutral tweets: " + str(len(tweets_all.loc[tweets_all.Sentiment == 'neutral'])))
#print(tweets_all.isnull().values.any())

tweets_all.to_csv(new_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
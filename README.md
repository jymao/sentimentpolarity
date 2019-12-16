# Sentiment Polarity Classification Using Machine Learning
Classifies Twitter posts as positive, neutral, or negative sentiment. Uses RandomForestClassifier and LinearSVC models from scikit-learn for classification. VADER from NLTK is used as a baseline.

datapreprocess.py is run first to extract relevant tweet data and merge datasets into one. The three datasets used are Sanders Analytics Twitter Sentiment Corpus, First GOP Debate Twitter Sentiment, and Twitter US Airline Sentiment. They are not included in this repository.

After creating the dataset, sentimentanalysis.py is run to train the models and print cross-validated F1 scores. The program will then accept custom string input for sentiment analysis.

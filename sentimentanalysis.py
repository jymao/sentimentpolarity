import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 
	'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
	'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
	'that', "that'll", 'these', 'those', 'am', 'be', 'been', 'being', 'having', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 
	'because', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'to', 'from', 'then', 'here', 
	'there', 'when', 'where', 'why', 'how', 'both', 'each', 'other', 'such', 'own', 'so', 'than', 'too', 'very', 's', 'will', 
	'd', 'll', 'm', 'o', 're', 've', 'y', 'becaus', 'have', 'ourselv', 'do', 'themselv', 'veri', 'whi', 'yourselv']

filename = "./data/tweets_all.csv"

# read data
tweets = pd.read_csv(filename)
print("Shape of tweets " + str(tweets.shape))
print("Positive tweets: " + str(len(tweets.loc[tweets.Sentiment == 'positive'])))
print("Negative tweets: " + str(len(tweets.loc[tweets.Sentiment == 'negative'])))
print("Neutral tweets: " + str(len(tweets.loc[tweets.Sentiment == 'neutral'])))

X = tweets.TweetText
y = tweets.Sentiment

# Helper class for preprocessing tweets before tokenization
# Lemmatizes, stems, and removes urls
class LemmaStemmer(object):
	def __init__(self):
		self.stemmer = SnowballStemmer("english")
		self.lemma = WordNetLemmatizer()
	def __call__(self, doc):
		return " ".join([self.stemmer.stem(self.lemma.lemmatize(word)) for word in doc.split() if not word.startswith('http://')])

# baseline: VADER
vader_analyzer = SentimentIntensityAnalyzer()

# split into train and test sets
#train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=5)

# 5-fold cross validation for model evaluation
kfold = KFold(n_splits=5, shuffle=True, random_state=5)

# avg f1 scores for each model over all iterations of cv
random_forest_f1 = [0,0,0]
svm_f1 = [0,0,0]
vader_f1 = [0,0,0]
num_features = 0 # avg number of features generated over all iterations of cv

current_cv_iteration = 1
for train_indices, test_indices in kfold.split(X):
	print("Cross validation iteration: " + str(current_cv_iteration) + " ----------")
	current_cv_iteration += 1

	train_X = X[train_indices]
	test_X = X[test_indices]
	train_y = y[train_indices]
	test_y = y[test_indices]

	# preprocess and extract features
	#vectorizer = CountVectorizer(ngram_range=(3,3))
	#vectorizer = CountVectorizer(analyzer='char',ngram_range=(5,5))
	vectorizer = CountVectorizer(preprocessor=LemmaStemmer(), analyzer='char', ngram_range=(5,5), stop_words=STOPWORDS)
	#vectorizer = CountVectorizer(preprocessor=LemmaStemmer(), stop_words=STOPWORDS, ngram_range=(3,3))
	train_X_vectorized = vectorizer.fit_transform(train_X)
	num_features += train_X_vectorized.shape[1]
	print("Shape of vectorized train_X: " + str(train_X_vectorized.shape))

	# oversampling train set to address class imbalance
	oversampler = RandomOverSampler(random_state=5)
	train_X_oversampled, train_y_oversampled = oversampler.fit_resample(train_X_vectorized, train_y)
	print("Shape of random oversampled train_X: " + str(train_X_oversampled.shape))

	# train models

	randomforest = RandomForestClassifier(n_estimators=100, random_state=5)
	randomforest.fit(train_X_oversampled, train_y_oversampled)
	print("Done with random forest training")

	svm = LinearSVC(max_iter=100000, random_state=5)
	svm.fit(train_X_oversampled, train_y_oversampled)
	print("Done with svm training")

	# predict test labels
	test_X_vectorized = vectorizer.transform(test_X)
	rf_pred_y = randomforest.predict(test_X_vectorized)
	svm_pred_y = svm.predict(test_X_vectorized)

	# baseline: VADER
	vader_pred_y = []
	for tweet in test_X:
		score = vader_analyzer.polarity_scores(tweet)
		if score['compound'] >= 0.05:
			vader_pred_y.append('positive')
		elif score['compound'] <= -0.05:
			vader_pred_y.append('negative')
		else:
			vader_pred_y.append('neutral')

	# print metrics
	print("----------------------------")
	print("RandomForest")
	print("Accuracy")
	print(accuracy_score(test_y, rf_pred_y))
	print("Confusion matrix")
	print(confusion_matrix(test_y, rf_pred_y, labels=['positive', 'neutral', 'negative']))
	print("Precision, recall, F1, support")
	results = precision_recall_fscore_support(test_y, rf_pred_y, average=None, labels=['positive', 'neutral', 'negative'])
	print(results)
	random_forest_f1[0] += results[2][0]
	random_forest_f1[1] += results[2][1]
	random_forest_f1[2] += results[2][2]

	print("----------------------------")
	print("SVM")
	print("Accuracy")
	print(accuracy_score(test_y, svm_pred_y))
	print("Confusion matrix")
	print(confusion_matrix(test_y, svm_pred_y, labels=['positive', 'neutral', 'negative']))
	print("Precision, recall, F1, support")
	results = precision_recall_fscore_support(test_y, svm_pred_y, average=None, labels=['positive', 'neutral', 'negative'])
	print(results)
	svm_f1[0] += results[2][0]
	svm_f1[1] += results[2][1]
	svm_f1[2] += results[2][2]

	print("----------------------------")
	print("VADER")
	print("Accuracy")
	print(accuracy_score(test_y, vader_pred_y))
	print("Confusion matrix")
	print(confusion_matrix(test_y, vader_pred_y, labels=['positive', 'neutral', 'negative']))
	print("Precision, recall, F1, support")
	results = precision_recall_fscore_support(test_y, vader_pred_y, average=None, labels=['positive', 'neutral', 'negative'])
	print(results)
	vader_f1[0] += results[2][0]
	vader_f1[1] += results[2][1]
	vader_f1[2] += results[2][2]

# Report average metrics on models
iterations = kfold.get_n_splits()
print("Avg num features: " + str(num_features / iterations))

print("Random Forest-----")
print("Pos F1: " + str(random_forest_f1[0] / iterations))
print("Neu F1: " + str(random_forest_f1[1] / iterations))
print("Neg F1: " + str(random_forest_f1[2] / iterations))
print("SVM---------------")
print("Pos F1: " + str(svm_f1[0] / iterations))
print("Neu F1: " + str(svm_f1[1] / iterations))
print("Neg F1: " + str(svm_f1[2] / iterations))
print("VADER-------------")
print("Pos F1: " + str(vader_f1[0] / iterations))
print("Neu F1: " + str(vader_f1[1] / iterations))
print("Neg F1: " + str(vader_f1[2] / iterations))

# live demo: accept custom strings to predict sentiment
# trained on all data
print("----------------------------")
print("Training demo models")

vectorizer = CountVectorizer(preprocessor=LemmaStemmer(), analyzer='char', ngram_range=(5,5), stop_words=STOPWORDS)
X_vectorized = vectorizer.fit_transform(X)
demo_analyzer = vectorizer.build_analyzer()
print("Shape of vectorized X: " + str(X_vectorized.shape))

oversampler = RandomOverSampler(random_state=5)
X_oversampled, y_oversampled = oversampler.fit_resample(X_vectorized, y)
print("Shape of random oversampled X: " + str(X_oversampled.shape))

randomforest = RandomForestClassifier(n_estimators=100, random_state=5)
randomforest.fit(X_oversampled, y_oversampled)
print("Done with random forest training")

svm = LinearSVC(max_iter=100000, random_state=5)
svm.fit(X_oversampled, y_oversampled)
print("Done with svm training")

print("----------------------------")
print("Demo: custom string input")
print("Press enter with no input to end")
print("String should be at least 5 characters")
demo = input("Input a string for sentiment analysis: ")

while demo != "":

	print(demo)

	demo_vectorized = vectorizer.transform([demo])
	print(demo_analyzer(demo))

	print("Random Forest: " + randomforest.predict(demo_vectorized)[0])
	print("SVM: " + svm.predict(demo_vectorized)[0])
	print("VADER: " + str(vader_analyzer.polarity_scores(demo)))

	demo = input("Input a string for sentiment analysis: ")










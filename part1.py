from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.corpus import stopwords
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def read_file_pandas(path):
    '''
    parameters
    ----------
    path: as STR

    returns
    -------
    Trump tweets
    '''
    with open(path, 'rb') as fin:
        data = pd.read_csv(path, sep='\t', names=['id', 'user_handle', 'text', 'time_stamp', 'device'])
        data["id"] = data["id"].astype("str")
    return data



def remove_noisy(text):
    """
    :param text:
    :return: clean_tweet
    """

    cleanData = text.replace('@', ' ')
    cleanData = cleanData.replace('#', ' ')

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    cleanData = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", cleanData)
    #remove stop words
    stop_free = " ".join([i for i in cleanData.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    cleanData= punc_free.replace('\d+', '')
    return cleanData


def is_Quote(text):
    '''
    parameters
    ----------
    text: as STR

    returns
    -------
    True if the text is  retweets
    '''
    if ((text[0]=="R" and text[1]=="T") or (text[0]=="r" and text[1]=="t") ):
        return 1
    return 0


def num_Exclamation(text):
    """
    :param text:
    :return:num_Exclamation
    """
    return sum(1 for c in text if c == "!")


def num_mentions(text):
    """

    :param text:
    :return: num_mentions
    """
    return sum(1 for c in text if c == "@")


def num_hashtags(text):
    """

    :param text:
    :return:
    """
    return sum(1 for c in text if c == "#")


def num_hastags_feature(data, numHastagsFeatureBool=True):
    """

    :param data:
    :param numHastagsFeature:
    :return:
    """
    if numHastagsFeatureBool:
        data["numHastags"] = data["text"].apply(lambda x: num_hashtags(x))

def qute_feature(dara, boolisQute=True):
    """

    :param dara:
    :param boolisQute:
    :return:
    """
    if boolisQute:
        dara["isQuote"] = map(is_Quote, dara["text"])

def num_mentions_feature(data, boolMentionsFeature=True):
    """

    :param data:
    :param boolMentionsFeature:
    :return:
    """

    if boolMentionsFeature:
        data["numMentions"] = data["text"].apply(lambda x: num_mentions(x))

def num_exlanation_feature(data, boolExlanationFeatures=True):
    """

    :param data:
    :param boolExlanationFeatures:
    :return:
    """
    if boolExlanationFeatures:
        data['Exclamation'] = data['text'].apply(lambda x: num_Exclamation(x))

def has_links_feature(data, boolFeaturesLinks=True):
    """

    :param data:
    :param boolFeaturesLinks:
    :return:
    """
    if boolFeaturesLinks:
        data['links'] = map(lambda text: (len(re.findall('https', text)) > 0), data.text)


def percentInCaps_func(text):
    """

    :param text:
    :return: percentInCaps
    """
    inCaps = sum(1 for c in text if c.isupper())
    inLowerCase = sum(1 for c in text if c.islower())
    if (inCaps + inLowerCase != 0):
        res= inCaps/float(inCaps + inLowerCase)
        return round(res, 2)
    return 0


punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~' # Excludes '#' and '@'
def percentPunctuation_func(text):

    """

    :param text:
    :return:percentPunctuation_func
    """
    return round(sum(1 for c in text if c in punctuation)/float(len(text)),2)



def tweet_length_feature(data, tweet_lengthBool=True):
    """

    :param data:
    :param tweet_lengthBool:
    :return: tweet_length
    """
    if (tweet_lengthBool):
        data['tweetLength'] = map(lambda tweet: len(tweet.split()), data['text'])


def time_hour_feature(data,time_featurs_hour=True ):
    """

    :param data:
    :param time_featurs_hour:
    :return:
    """
    if (time_featurs_hour):
        data['hourOfDay'] = pd.to_datetime(data.time_stamp, errors='coerce').dt.hour

def time_day_feature(data,time_featurs_day=True ):
    """

    :param data:
    :param time_featurs_day:
    :return:
    """
    if (time_featurs_day):
        data['dayOfWeek']= pd.to_datetime(data.time_stamp, errors='coerce').dt.dayofweek


def time_weekend_feature(data,time_featurs_Weekend=True ):
    """

    :param data:
    :param time_featurs_Weekend:
    :return:
    """
    if (time_featurs_Weekend):
        data['isWeekend'] = map(lambda x: (x == 5 or x == 6), data.dayOfWeek)


def percent_punctuation_feature(data, textFeaturesPercentPunctuation=True):
    """

    :param data:
    :param textFeaturesPercentPunctuation:
    :return:
    """
    if textFeaturesPercentPunctuation:
        data["percentPunctuation"] = data["text"].apply(lambda x: percentPunctuation_func(x))


def percent_caps_feature(data, textFeaturesPercentInCaps=True):
    """

    :param data:
    :param textFeaturesPercentInCaps:
    :return:
    """
    if textFeaturesPercentInCaps:
        data["percentInCaps"] = 2*data["text"].apply(lambda x: percentInCaps_func(x))

def tfIdf_features(data):
    """

    :param data:
    :return:data with
    """
    data['text'] = data['text'].apply(lambda tweet: remove_noisy(tweet))
    vec = TfidfVectorizer()
    vectfidf = vec.fit_transform(data['text']).toarray()
    cols = vec.get_feature_names()
    cols = [str(x) for x in cols]
    vec_df = pd.DataFrame(vectfidf, columns=cols, index=data.index)
    data = data.merge(vec_df, right_index=True, left_index=True)
    return data



def model_accuracy(trained_model, features, targets):
    """
    Get the accuracy score of the model
    :param trained_model:
    :param features:
    :param targets:
    :return:accuracy_score
    """
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score




def add_fetures_func(data, boolisQute=True, boolHastagsFeature=True, boolMentionsFeature=True, bool_features_exlanation=True, boolFeaturesLinks= True, boolFeaturesPercentPunctuation=True, boolFeaturesPercentInCaps=True, boolTweetLength=True, boolTimeFeatursHoure=True, boolTimeFeatursDay=True, boolTimeFeatursWeekend=True):

    """
    :param path:
    :param boolisQute:
    :param boolHastagsFeature:
    :param boolMentionsFeature:
    :param bool_features_exlanation:
    :param boolFeaturesLinks:
    :param boolFeaturesPercentPunctuation:
    :param boolFeaturesPercentInCaps:
    :param boolTweetLength:
    :param boolTimeFeatursHoure:
    :param boolTimeFeatursDay:
    :param boolTimeFeatursWeekend:
    :return:

    """

    #check if it is Qute
    qute_feature(data, boolisQute)
    #check muber of Hastags
    num_hastags_feature(data, boolHastagsFeature)
    #check muber of Mentions
    num_mentions_feature(data, boolMentionsFeature)
    #numExlanationFeature
    num_exlanation_feature(data, bool_features_exlanation)
    #hasLinksFeature
    has_links_feature(data, boolFeaturesLinks)
    #percentPunctuationFeature
    percent_punctuation_feature(data, boolFeaturesPercentPunctuation)
    #percentInCapsFeature
    percent_caps_feature(data, boolFeaturesPercentInCaps)
    #tweetLengthFeature
    tweet_length_feature(data,boolTweetLength)
    #Hour Feature
    time_hour_feature(data, boolTimeFeatursHoure)
    # day Feature
    time_day_feature(data, boolTimeFeatursDay)
    # Weekend Feature
    time_weekend_feature(data, boolTimeFeatursWeekend)
    return data



def preprocessing(data, boolisQute=True, boolHastagsFeature=True, boolMentionsFeature=True, bool_features_exlanation=True, boolFeaturesLinks= True, boolFeaturesPercentPunctuation=True, boolFeaturesPercentInCaps=True, boolTweetLength=True, boolTimeFeatursHoure=True, boolTimeFeatursDay=True, boolTimeFeatursWeekend=True):
    """
    :param path:
    :param boolisQute:
    :param boolHastagsFeature:
    :param boolMentionsFeature:
    :param bool_features_exlanation:
    :param boolFeaturesLinks:
    :param boolFeaturesPercentPunctuation:
    :param boolFeaturesPercentInCaps:
    :param boolTweetLength:
    :param boolTimeFeatursHoure:
    :param boolTimeFeatursDay:
    :param boolTimeFeatursWeekend:
    :return:

    """
    #get only android and iphone data
    dataTrain = data.loc[data['device'].isin(['android', 'iphone'])]
    #remove obama Tweets
    dataTrain=dataTrain.drop(dataTrain[(dataTrain['user_handle']== "POTUS") & (dataTrain['time_stamp']>'2017-01-20')].index)
    #get twits before 2017-04-01
    dataTrain=dataTrain[pd.to_datetime(dataTrain['time_stamp'], errors='coerce')<'2017-04-01']

    dataTrain=dataTrain.drop(dataTrain[(dataTrain['user_handle']== "POTUS")].index)
    dataTrain = dataTrain.drop(dataTrain[(dataTrain['user_handle'] == "PressSec")].index)
    dataTest = data[~data.id.isin(dataTrain['id'])]

    dataTrain = dataTrain.drop(['id'], axis=1)

    dataTrain= add_fetures_func(dataTrain, boolisQute,boolHastagsFeature, boolMentionsFeature, bool_features_exlanation,
                                boolFeaturesLinks, boolFeaturesPercentPunctuation, boolFeaturesPercentInCaps,boolTweetLength,
                                boolTimeFeatursHoure, boolTimeFeatursDay, boolTimeFeatursWeekend)

    dataTrain= tfIdf_features(dataTrain)
    trian_X = dataTrain.drop([ 'text', 'device', 'time_stamp','user_handle'],axis=1)

    trian_y = dataTrain['device']
    return trian_X, trian_y, dataTest




def classification(path, boolisQute=True, boolHastagsFeature=True, boolMentionsFeature=True,
                   bool_features_exlanation=True, boolFeaturesLinks= True, boolFeaturesPercentPunctuation=True,
                   boolFeaturesPercentInCaps=True, boolTweetLength=True, boolTimeFeatursHoure=True, boolTimeFeatursDay=True,
                   boolTimeFeatursWeekend=True):
    """

    :param path:
    :param boolisQute:
    :param boolHastagsFeature:
    :param boolMentionsFeature:
    :param bool_features_exlanation:
    :param boolFeaturesLinks:
    :param boolFeaturesPercentPunctuation:
    :param boolFeaturesPercentInCaps:
    :param boolTweetLength:
    :param boolTimeFeatursHoure:
    :param boolTimeFeatursDay:
    :param boolTimeFeatursWeekend:
    :return:
    """
    data = read_file_pandas(path)
    X, Y, test = preprocessing(data, boolisQute, boolHastagsFeature, boolMentionsFeature, bool_features_exlanation,
                               boolFeaturesLinks, boolFeaturesPercentPunctuation, boolFeaturesPercentInCaps, boolTweetLength,
                               boolTimeFeatursHoure, boolTimeFeatursDay, boolTimeFeatursWeekend)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    names=["LogisticRegression", "svm-linear",  "DecisionTreeClassifier","RandomForestClassifier",  "AdaBoostClassifier",
           "GaussianNB"]
    #the differnet classifiers that will be checked
    classifiers = [
       LogisticRegression(),
        #svm.SVC(kernel='linear'),
        svm.SVC(kernel="linear"),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB()]
    for name, clf in zip(names, classifiers):
        print(name)
        clf.fit(X_train, y_train)
        res = clf.predict(X_test)
        accuracy_test = accuracy_score(y_test, res)
        print 'accuracy_test: ', accuracy_test
        print(metrics.classification_report(y_test, res))



def main(path):
    """

    :param path:
    :return:
    """
    classification(path)


path="tweets.tsv"
main(path)
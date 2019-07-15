from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def nb_news():
    news = fetch_20newsgroups(subset="all", categories=['rec.sport.hockey', 'rec.motorcycles'])

    x_train, x_test, y_train, y_test = train_test_split(news.data,news.target)

    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)

    score = estimator.score(x_test, y_test)
    print("score：\n", score)

    sentence = input("Enter some text: ")
    sentence_x = transfer.transform([sentence])
    y_predict = estimator.predict(sentence_x)
    print("y_predict:\n", y_predict)
 
    return None

nb_news()
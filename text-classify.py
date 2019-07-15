from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def text_classify():
    data = [ "Help me impress the girl of my dreams!", 
             "How do you measure ingredients like butter in cups?",
             "Tips on making fried rice", 
             "immutability in javascript. It has a declarative approach of programming, which means that you focus on describing what your program must accomplish", 
             "Facing a Programming Problem. Everybody has encountered it, the programming problem that makes NO sense. This problem has no fix, it just cannot be done",
             " 5 Uses for the Spread Operator. The spread operator is a favorite of JavaScript developers. It's a powerful piece of syntax that has numerous applications."]
    target = [ 0,0,0,1,1,1 ]
    x_train, x_test, y_train, y_test = train_test_split(data,target)

    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)

    score = estimator.score(x_test, y_test)
    print("score：\n", score)

    sentence = input("Enter some text: ")
    sentence_x = transfer.transform([sentence])
    y_predict = estimator.predict(sentence_x)
    print("y_predict: ", y_predict)
 
    return None

text_classify()
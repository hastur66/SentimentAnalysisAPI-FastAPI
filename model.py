from re import M
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
# from util import plot_roc

class NLPModel(object):

    def __init__(self) -> None:
        """NLP model
        Attributes:
            clf: classifier
            vectorizer: TFIDF vectorizer
        """
        self.clf = MultinomialNB()
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, X):
        """Fit Vectorizer with text
        """
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """Transform the text to TFIDF matrix
        """
        X_transform = self.vectorizer.transform(X)
        return X_transform

    def train(self, X, y):
        """Train classifier
        """
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Return the prediction class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='lib/models/TFIDFVectorizer.pkl'):
        """Save the vectorizer in pickle format
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer in {}".format(path))
    
    def pickle_clf(self, path='lib/models/SentimentClassifier.pkl'):
        """Save the classifier model in pickle format
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifer in {}".format(path))

    def plot_roc(self, X, y, size_x, size_y):
        """Plot the ROC curve for X_test and y_test.
        """
        # plot_roc(self.clf, X, y, size_x, size_y)
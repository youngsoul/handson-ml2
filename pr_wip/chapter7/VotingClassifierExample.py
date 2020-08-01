# page 192

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# True/soft - will use predict_proba predict class with highest probablity across all of the classifiers
# False/hard - will pick the class with the most votes
probability = False
voting='hard'
if __name__ == '__main__':

    X, y = make_moons(n_samples=500, noise=0.25)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC(probability=False) # False does not predict a probability but a 0/1 classification

    voting_clf = VotingClassifier(estimators=[
        ('lr', log_clf),
        ('rf', rnd_clf),
        ('svc', svm_clf)
        ],
        voting='hard'
    )

    # voting_clf.fit(X_train, y_train)
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



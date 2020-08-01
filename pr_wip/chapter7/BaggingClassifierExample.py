# page 194
# page 192

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    X, y = make_moons(n_samples=1000, noise=0.25)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=500,
        max_samples=100,
        bootstrap=True, # with replacement
        n_jobs=-1
    )

    bag_clf.fit(X_train, y_train)

    y_pred = bag_clf.predict(X_test)

    print("BagClassifier(DecisionTreeClassifier): ", accuracy_score(y_test, y_pred))

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("DecisionTreeClassifier: ", accuracy_score(y_test, y_pred))

    bag_clf2 = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=500,
        max_samples=100,
        bootstrap=True, # with replacement
        n_jobs=-1,
        oob_score=True
    )
    bag_clf2.fit(X_train, y_train)
    print(bag_clf2.oob_score_)

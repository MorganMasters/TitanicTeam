import pandas as pd
import sklearn as skl
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn import svm
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


def data_cleaner(data):
    """
    Feature extraction pipeline.

    :param data: Pandas DataFrame, loaded from either train.csv or test.csv from the Kaggle Titanic challenge dataset
    :return: data: Numpy array with all categorical data one-hot encoded, and the remainder column-wise scaled to zero
                mean and unit variance.
    """
    # sex and pclass to dummies
    label = LabelEncoder()
    sex1H = pd.get_dummies(label.fit_transform(data.Sex))
    pclass1H = pd.get_dummies(label.fit_transform(data.Pclass))

    # add family size feature
    data['Family Size'] = data.Parch + data.SibSp + 1

    # featurize name into trainable titles indicating social class
    data['Title'] = data.Name.str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    title_dictionary = {
        "Capt": "Military",
        "Col": "Military",
        "Major": "Military",
        "Jonkheer": "Noble",
        "Don": "Noble",
        "Sir": "Noble",
        "Dr": "Misc",
        "Rev": "Misc",
        "the Countess": "Noble",
        "Mme": "Common",
        "Mlle": "Young",
        "Ms": "Common",
        "Mr": "Common",
        "Mrs": "Common",
        "Dona": "Noble",
        "Miss": "Young",
        "Master": "Young",
        "Lady": "Noble"}
    data.Title = data.Title.map(title_dictionary)
    title1H = pd.get_dummies(label.fit_transform(data.Title))
    data.drop('Name', axis=1, inplace=True)

    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    emb1H = pd.get_dummies(label.fit_transform(data['Embarked']))

    drops = ['Sex', 'Pclass', 'Title', 'Embarked']
    data.drop(drops, axis=1, inplace=True)

    dummy_list = [sex1H, pclass1H, title1H, emb1H]
    for item in dummy_list:
        data = pd.concat([data, item], axis=1)

    return data


model = 'svm'
scale = 'standard'

train = pd.read_csv("inputs/train.csv")
test = pd.read_csv("inputs/test.csv")
ids = test.PassengerId
target = train.Survived
train.drop('Survived', axis=1, inplace=True)
drops = ['PassengerId', 'Cabin', 'Ticket']
train.drop(drops, axis=1, inplace=True)
test.drop(drops, axis=1, inplace=True)

clean_tr = data_cleaner(train)
clean_te = data_cleaner(test)

imp = IterativeImputer(max_iter=10, random_state=0)
clean_tr = imp.fit_transform(clean_tr.astype('float64'))
clean_te = imp.fit_transform(clean_te.astype('float64'))

if scale == 'standard':
    # Scale non-categorical values for easier model use
    for n in range(5):
        clean_tr[:, n] = skl.preprocessing.scale(clean_tr[:, n])
        clean_te[:, n] = skl.preprocessing.scale(clean_te[:, n])

if model == 'svm':
    clf = svm.NuSVC(probability=True)
if model == 'rfc':
    clf = RandomForestClassifier(max_depth=8, random_state=0)
bc = BaggingClassifier(base_estimator=clf, n_estimators=100)

scores = skl.model_selection.cross_val_score(clf, clean_tr, target, cv=5)
print(scores)
print(np.mean(scores))

# Run selected model
bc.fit(clean_tr, target)
y = bc.predict(clean_te)

resultFile = open("outputs/result.csv", 'w')
resultFile.write("PassengerID,Survived\n")
for i in range(0, len(y)):
    resultFile.write(str(ids[i]) + "," + str(y[i]) + "\n")
resultFile.close()

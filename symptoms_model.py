import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

def symptoms_prediction(symptoms):
    dataset1 = pd.read_csv("Training.csv")
    dataset2 = pd.read_csv("Testing.csv")
    dataset = dataset1.append(dataset2, ignore_index = True)

    X = dataset.iloc[:,0:132]
    Y = dataset.iloc[:,132:133]

    kf = KFold(n_splits=10)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        dtree = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.011)
        Feature_train, Feature_test=X.iloc[train_index],X.iloc[test_index]
        Label_train, Label_test=Y.iloc[train_index],Y.iloc[test_index]
        dtree.fit(Feature_train, Label_train.values.ravel())


    return dtree.predict(symptoms)

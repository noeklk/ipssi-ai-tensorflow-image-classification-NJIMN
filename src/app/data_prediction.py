import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

def get_data():
    data = pd.read_csv('./StudentsPerformance.csv')

    data.columns = "gender","race","parental_edu","lunch","test_prep","math","reading","writing"

    return data

def notes_prediction(target, input_data):
    
    data = get_data()

    X = data[["math", "reading", "writing"]]
    y = data[target]

    # prediction
    model = DecisionTreeClassifier()
    model.fit(X, y)

    predictions = model.predict([input_data])
    predictions

    return predictions


def character_type_prediction(input_data):
    
    # preprocesssing the data
    data = get_data()

    LE = preprocessing.LabelEncoder()

    data.columns = "gender", "race", "parental_edu", "lunch", "test_prep", "math", "reading",   "writing"
    data = data.apply(LE.fit_transform)

    X = data[["gender", "race", "lunch"]]
    y = data[["math", "reading", "writing"]]

    # prediction
    model = DecisionTreeClassifier()
    model.fit(X, y)

    predictions = model.predict([input_data])

    predictions = pd.DataFrame(predictions)

    new_df = pd.concat([predictions[0], predictions[1], predictions[2]], axis=1)
    new_df.columns = "Mathématiques", "Lecture", "Ecriture"

    return new_df


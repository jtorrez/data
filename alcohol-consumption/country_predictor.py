import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def prep_X_y(df):
    y = df['country'].values
    X = df.drop('country', axis=1).values
    return X, y

def fit_and_predict(X, y, user_data):
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X, y)
    return rf.predict(user_data)

def predict_country():
    #The US serving size for one serving of alcohol in liters is 0.0177
    ss = 0.0177

    print "\nLet's figure out which country your drinking habits belong to...\n"
    print "First, let's look at how much beer you drink:"

    beer = float(raw_input("How many servings of beer do you drink per week? "))
    beer *= 52

    print "\nNext, let's look at how much hard alcohol (spirits) you drink:"
    spirit = float(raw_input("How many servings of hard alcohol do you drink per week? "))
    spirit *= 52

    print "\nNext up, wine time."

    wine = float(raw_input("How many servings of wine do you drink per week? "))
    wine *= 52
    total_L_alc = (beer * ss + spirit * ss + wine * ss)

    print "\nOk, give me a second..."
    print "..."
    print "..."
    print "..."

    user_data = np.array([[beer, spirit, wine, total_L_alc]])
    data = load_data('drinks.csv')
    X, y = prep_X_y(data)
    prediction = fit_and_predict(X, y, user_data)

    print "\nLooks like you belong in {}!".format(prediction[0])
    print "\nBetter pack your things!\n"

if __name__ == '__main__':
    predict_country()

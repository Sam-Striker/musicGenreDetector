import numpy as np 
import pandas as pd 

df = pd.read_csv('music.csv')
#print(df) 

# df.describe() 
# print(df.duplicated)
# df.drop_duplicates
                          # cleaning dataset
                          ###################
# replace nar rows with the median
# print(df.columns)
# df['age'] = df['age'].fillna(df['age'].mean())
# df
# df['age'].fillna('True', inplace=True)

X = df[['age','gender']]
Y = df[['genre']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)  #train model

prediction = model.predict(X_test)

from sklearn.metrics import accuracy_score
import joblib

joblib.dump(model,'music_trained_model.joblib') #save model as file 

score = accuracy_score(Y_test, prediction)
print(score)
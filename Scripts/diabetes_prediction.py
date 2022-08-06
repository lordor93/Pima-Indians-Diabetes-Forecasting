################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################

import joblib
import pandas as pd
from Scripts.diabetes_pipeline import *

df = pd.read_csv("C:/Users/emrek/Desktop/DataSets/diabetes.csv")

random_user = df.sample(1, random_state=45)

new_model = joblib.load("../voting_clf.pkl")


X, y = diabetes_data_prep(df)

random_user = X.sample(1, random_state=50)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)

import joblib
from Scripts.diabetes_pipeline import *

################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_csv("C:/Users/emrek/Desktop/DataSets/diabetes.csv")
    X, y = diabetes_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "../voting_clf.pkl")
    return voting_clf

# In order to run this file in terminal we can write main class.
if __name__ == "__main__":
    print("Process was initiated.")
    main()


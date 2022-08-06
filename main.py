

import joblib
from 

################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_csv("C:/Users/emrek/Desktop/VBO BootCamp/Miuul/DataSets/diabetes.csv")
    X, y = diabetes_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "../voting_clf.pkl")
    return voting_clf

# komut satırından calistirmak icin main class yazılır.
if __name__ == "__main__":
    print("İşlem başladı")
    main()


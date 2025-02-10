from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import wandb

def read_data():
    crops = pd.read_csv("data/soil_measures.csv")
    return crops


def split_data(crops):
    X = crops.drop("crop",  axis=1)
    y = crops["crop"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
        )
    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    scores = {}
    table = wandb.Table(columns=["Feature", "F1-Score"])
    
    for feature in ["N", "P", "K", "ph"]:
        log_reg = LogisticRegression(multi_class="multinomial", solver='lbfgs')
        log_reg.fit(X_train[[feature]], y_train)
        y_pred = log_reg.predict(X_test[[feature]])
        f1 = metrics.f1_score(y_test, y_pred, average="weighted")
        scores[feature] = f1
        table.add_data(feature, f1)

    # Best predictive
    best_feature = max(scores, key=scores.get)
    best_predictive_feature = {best_feature: scores[best_feature]}
    wandb.log(best_predictive_feature)
    wandb.log({"feature_scores": table})
    return scores, best_feature


def visualize_best_feature(scores, best_feature):
    plt.figure(figsize=(8, 5))
    plt.bar(scores.keys(), scores.values())
    plt.title("Feature Importance")
    plt.xlabel("Soil Characteristics")
    plt.ylabel("F1-Score")
    wandb.log({"feature_importance": wandb.Image(plt)})


def main():
        # Load the dataset
        crops = read_data()

        # Split data
        X_train, X_test, y_train, y_test = split_data(crops)

        # Performance
        scores, best_feature = train(X_train, X_test, y_train, y_test)

        #Visualization
        visualize_best_feature(scores, best_feature)


if __name__ == "__main__":
    main()
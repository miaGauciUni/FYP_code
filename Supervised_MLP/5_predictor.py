import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingUsingMLP:
    def __init__(self, file_name, stock_name):
        self.file_name = file_name
        self.stock_name = stock_name
        self.data = pd.read_csv(file_name)
        print(f"Data loaded from {file_name} for {stock_name}.")

    def save_data_as_pickle(self, pickle_file_name):
        data_array = self.data.to_numpy()
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(data_array, f)

    def split_data(self, window):
        cols = []
        for prefix in ["Closing", "EMA_5", "RSI", "MACD", "BB_upper", "BB_lower"]:
            for i in range(window, 0, -1):  # e.g. 5 to 1
                cols.append(f"{prefix}_{i}")
        X = self.data[cols]
        y = self.data['Action']
        return train_test_split(X, y, test_size=0.4, random_state=42)

    def train_mlp(self, X_train, y_train, hidden_layers, max_iter, alpha, learning_rate_init, activation, solver):
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            activation=activation,
            solver=solver,
        )
        mlp.fit(X_train, y_train)
        return mlp

    def evaluate_model(self, mlp, X_train, X_test, y_train, y_test):
        y_pred_test = mlp.predict(X_test)
        y_pred_train = mlp.predict(X_train)

        labels = [-1, 0, 1, 2]
        cm = confusion_matrix(y_test, y_pred_test, labels=labels)

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Pred {l}' for l in labels],
                    yticklabels=[f'Actual {l}' for l in labels],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix for {self.stock_name}')
        plt.tight_layout()
        fig.savefig(f"confusion_matrix_{self.stock_name}.png", dpi=600)
        plt.close()

        print(f"Test Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
        print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        print(f"Test Precision: {precision_score(y_test, y_pred_test, average='weighted', zero_division=1):.4f}")
        print(f"Test Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")

    def save_model_as_pickle(self, mlp, model_file_name):
        with open(model_file_name, 'wb') as f:
            pickle.dump(mlp, f)


# === Master controller for multiple stocks and windows ===
def run_all(window_sizes, stock_list):
    for window in window_sizes:
        for stock in stock_list:
            try:
                input_csv = f"5_processed_with_actions/{stock}_{window}day_tech_action.csv"
                data_pkl = f"5_processed_with_actions/{stock}_{window}day_tech_action.pkl"
                model_pkl = f"6_stock_models/{stock}_MLP_model_{window}day.pkl"

                trainer = TrainingUsingMLP(input_csv, stock)
                trainer.save_data_as_pickle(data_pkl)

                X_train, X_test, y_train, y_test = trainer.split_data(window)
                mlp = trainer.train_mlp(X_train, y_train,
                                        hidden_layers=(64,32, 16, 8),
                                        max_iter=5000,
                                        alpha=1e-4,
                                        learning_rate_init=5e-4,
                                        activation='tanh',
                                        solver='lbfgs')
                trainer.evaluate_model(mlp, X_train, X_test, y_train, y_test)
                trainer.save_model_as_pickle(mlp, model_pkl)

                print(f"Completed: {stock} | Window {window} days\n")

            except FileNotFoundError:
                print(f"[!] File missing for {stock} with {window}-day window — Skipped.\n")
            except Exception as e:
                print(f"[!] Error processing {stock} ({window}-day): {e}\n")


if __name__ == '__main__':
    window_sizes = [5]
    stock_list = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"]  # Update with your actual 7 tickers
    run_all(window_sizes, stock_list)

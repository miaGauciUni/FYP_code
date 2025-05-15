import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingUsingMLP:
    def __init__(self, file_name, stock_name):
        """
        Initializes the class with the CSV file name, loads the data into a DataFrame,
        and sets the stock name for identification.
        """
        self.file_name = file_name
        self.stock_name = stock_name
        self.data = pd.read_csv(file_name)
        print(f"Data loaded from {file_name} for {stock_name}.")

    def save_data_as_pickle(self, pickle_file_name):
        """
        Saves the dataset as a pickle file for later use.
        """
        data_array = self.data.to_numpy()
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(data_array, f)
        print(f"Data saved to {pickle_file_name} as a pickle.")

    def split_data(self):
        """
        Splits the data into features (X) and target (y) and then splits into training and test sets.
        X now includes technical indicators as well as the past 5 days' closing prices.
        """
        # Define all technical indicators and closing prices as features
        X = self.data[['Closing_3', 'Closing_2', 'Closing_1',
            'EMA_5_3', 'EMA_5_2', 'EMA_5_1',
            'RSI_3', 'RSI_2', 'RSI_1',
            'MACD_3', 'MACD_2', 'MACD_1',
            'BB_upper_3', 'BB_upper_2', 'BB_upper_1',
            'BB_lower_3', 'BB_lower_2', 'BB_lower_1']]

        y = self.data['Action']  # Target variable remains the action labels


        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        print(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}.")
        return X_train, X_test, y_train, y_test

    def train_mlp(self, X_train, y_train,
                  hidden_layers, max_iter, alpha,
                  learning_rate_init, activation,
                  solver='adam', random_state=42):
        """
        Trains an MLP classifier using the specified hyperparameters.
        """
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            activation=activation,
            solver=solver,
            random_state=random_state
        )
        mlp.fit(X_train, y_train)
        print(f"MLP model trained for {self.stock_name} with params: "
              f"{hidden_layers}, {max_iter}, {alpha}, {learning_rate_init}, "
              f"{activation}, solver={solver}.")
        return mlp

    def evaluate_model(self, mlp, X_train, X_test, y_train, y_test):
        """
        Evaluates the MLP model on both the training and test sets,
        prints metrics, and plots + saves the test confusion matrix.
        """
        y_pred_test  = mlp.predict(X_test)
        y_pred_train = mlp.predict(X_train)

        # ——— Confusion matrix ———
        labels = [-1, 0, 1, 2]  # loss, hold, buy, strong buy
        cm = confusion_matrix(y_test, y_pred_test, labels=labels)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[f'Pred {l}' for l in labels],
            yticklabels=[f'Actual {l}' for l in labels],
            ax=ax
        )
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(f'Confusion Matrix for {self.stock_name}')
        plt.tight_layout()
        # Save at 600 DPI
        fig.savefig(f"confusion_matrix_{self.stock_name}.png", dpi=600, bbox_inches='tight')
        plt.show()

        # ——— Metrics ———
        recall_test    = recall_score(y_test,  y_pred_test, average='weighted')
        accuracy_test  = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        print(f"\nTest Recall for {self.stock_name}:    {recall_test:.4f}")
        print(f"Test Accuracy for {self.stock_name}:  {accuracy_test:.4f}")
        print(f"Test Precision for {self.stock_name}: {precision_test:.4f}")
        print(f"Train Accuracy for {self.stock_name}: {accuracy_train:.4f}")

    def save_model_as_pickle(self, mlp, model_file_name):
        """
        Saves the trained MLP model as a pickle file.
        """
        with open(model_file_name, 'wb') as f:
            pickle.dump(mlp, f)
        print(f"MLP model for {self.stock_name} saved to {model_file_name}.")

    def run(self, data_pickle_file_name, model_pickle_file_name,
            hidden_layers, max_iter, alpha,
            learning_rate_init, activation,
            solver='adam', random_state=42):
        """
        Executes the full pipeline.
        """
        self.save_data_as_pickle(data_pickle_file_name)
        X_train, X_test, y_train, y_test = self.split_data()
        mlp = self.train_mlp(
            X_train, y_train,
            hidden_layers, max_iter, alpha,
            learning_rate_init, activation,
            solver, random_state
        )
        self.evaluate_model(mlp, X_train, X_test, y_train, y_test)
        self.save_model_as_pickle(mlp, model_pickle_file_name)


# Define separate functions for each stock with their specific hyperparameters
def train_microsoft():
    mlp_training = TrainingUsingMLP(
        "5_processed_with_actions/AAPL_3day_tech_action.csv", "AAPL"
    )
    mlp_training.run(
        "5_processed_with_actions/AAPL_3day_tech_action.pkl",
        "6_stock_models/AAPL_MLP_model.pkl",
        hidden_layers=(32, 16, 8),  # larger capacity
        max_iter=4000,                 # plenty of iterations
        alpha=1e-4,                    # moderate L2 penalty
        learning_rate_init=0.0001,      # unused by lbfgs but required by signature
        activation='relu',             # efficient gradients
        solver='lbfgs',                # more stable on medium data
        #random_state=42                # reproducible results
    )



def train_google():
    mlp_training = TrainingUsingMLP(
        "5_processed_with_actions/MSFT_3day_tech_action.csv", "MSFT")
    mlp_training.run(
        "5_processed_with_actions/MSFT_3day_tech_action.pkl",
        "6_stock_models/MSFT_MLP_model.pkl",
        hidden_layers=(32, 16, 8),  # larger capacity
        max_iter=4000,  # plenty of iterations
        alpha=1e-4,  # moderate L2 penalty
        learning_rate_init=0.0001,  # unused by lbfgs but required by signature
        activation='relu',  # efficient gradients
        solver='lbfgs',  # more stable on medium data
        #random_state=42
    )


def train_apple():
    mlp_training = TrainingUsingMLP(
        "5_processed_with_actions/GOOGL_3day_tech_action.csv", "GOOGLE")
    mlp_training.run(
        "5_processed_with_actions/GOOGL_3day_tech_action.pkl",
        "6_stock_models/GOOGL_MLP_model.pkl",
        hidden_layers=(32, 16, 8),  # larger capacity
        max_iter=4000,  # plenty of iterations
        alpha=1e-4,  # moderate L2 penalty
        learning_rate_init=0.0001,  # unused by lbfgs but required by signature
        activation='relu',  # efficient gradients
        solver='lbfgs',  # more stable on medium data
        #random_state=42
    )


def train_nvidia():
    mlp_training = TrainingUsingMLP(
        "5_processed_with_actions/META_3day_tech_action.csv", "META")
    mlp_training.run(
        "5_processed_with_actions/META_3day_tech_action.pkl",
        "6_stock_models/META_MLP_model.pkl",
        hidden_layers=(32, 16, 8),  # larger capacity
        max_iter=4000,  # plenty of iterations
        alpha=1e-4,  # moderate L2 penalty
        learning_rate_init=0.0001,  # unused by lbfgs but required by signature
        activation='relu',  # efficient gradients
        solver='lbfgs',  # more stable on medium data
        #random_state=42
    )


def train_amazon():
    mlp_training = TrainingUsingMLP(
        "5_processed_with_actions/AMZN_3day_tech_action.csv", "Amazon")
    mlp_training.run(
        "5_processed_with_actions/AMZN_3day_tech_action.pkl",
        "6_stock_models/AMZN_MLP_model.pkl",
        hidden_layers=(32, 16, 8),  # larger capacity
        max_iter=4000,  # plenty of iterations
        alpha=1e-4,  # moderate L2 penalty
        learning_rate_init=0.0001,  # unused by lbfgs but required by signature
        activation='relu',  # efficient gradients
        solver='lbfgs',  # more stable on medium data
        #random_state=42
    )


if __name__ == '__main__':
    # Train each stock individually with its specific hyperparameters
    train_microsoft()
    train_google()
    train_apple()
    train_nvidia()
    train_amazon()
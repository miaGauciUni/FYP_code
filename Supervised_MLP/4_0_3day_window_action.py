import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ProcessingData:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name)
        self.actions_df = None
        print(f"Data loaded from {file_name}.")

    def calculate_price_differences(self):
        self.df['Price_Difference'] = self.df['Closing'].diff()
        print("Price differences calculated.")
        self.df.to_csv(self.file_name, index=False)
        print(f"Price differences saved to {self.file_name}.")

    def calculate_quantiles(self):
        price_differences = self.df['Price_Difference'].dropna()
        quantile1 = np.quantile(price_differences[price_differences > 0], 0.45)
        quantile2 = np.quantile(price_differences[price_differences > 0], 0.76)
        print(f"Quantiles calculated:\nBuy (q1): {quantile1}\nStrong Buy (q2): {quantile2}")
        return quantile1, quantile2

    def classify_differences(self, q1, q2):
        print(f"Cutoff points for classification:\nHold (0): {0}\nBuy (1): {q1}\nStrong Buy (2): {q2}")

        self.df['Classification'] = np.select(
            [
                self.df['Price_Difference'] < 0,
                (self.df['Price_Difference'] >= 0) & (self.df['Price_Difference'] < q1),
                (self.df['Price_Difference'] >= q1) & (self.df['Price_Difference'] < q2),
                (self.df['Price_Difference'] >= q2)
            ],
            [-1, 0, 1, 2],
            default=-1
        )

        counts = self.df['Classification'].value_counts()
        print("\nClassification Counts:")
        for action, count in counts.items():
            action_label = 'Loss Action (-1)' if action == -1 else f'Action {action}'
            print(f"{action_label}: {count}")

    def evaluate_last_3_days(self):
        actions = []

        for i in range(3, len(self.df)):
            Closings = self.df['Closing'].iloc[i - 3:i].values
            ema_5 = self.df['EMA_5'].iloc[i - 3:i].values
            rsi = self.df['RSI'].iloc[i - 3:i].values
            macd = self.df['MACD'].iloc[i - 3:i].values
            bb_upper = self.df['BB_upper'].iloc[i - 3:i].values
            bb_lower = self.df['BB_lower'].iloc[i - 3:i].values

            action = self.determine_action(Closings[-1], Closings[:-1])
            actions.append([
                *Closings, *ema_5, *rsi, *macd, *bb_upper, *bb_lower, action
            ])

        self.actions_df = pd.DataFrame(actions, columns=[
            'Closing_3', 'Closing_2', 'Closing_1',
            'EMA_5_3', 'EMA_5_2', 'EMA_5_1',
            'RSI_3', 'RSI_2', 'RSI_1',
            'MACD_3', 'MACD_2', 'MACD_1',
            'BB_upper_3', 'BB_upper_2', 'BB_upper_1',
            'BB_lower_3', 'BB_lower_2', 'BB_lower_1',
            'Action'
        ])

        # Update the Action column directly from Classification values
        self.actions_df['Classification'] = self.df['Classification'].iloc[3:].values

        self.actions_df.to_csv("5_processed_with_actions/AAPL_3day_tech_action.csv", index=False)
        print("Actions with technical indicators saved to AAPL_3day_tech_action.csv.")

    def determine_action(self, current_Closing, previous_Closings):
        price_diff = current_Closing - previous_Closings[-1]

        if price_diff < 0:
            return -1  # Loss action
        elif price_diff >= 0 and price_diff < 1:
            return 0  # Hold action
        elif price_diff >= 1 and price_diff < 2:
            return 1  # Buy action
        else:
            return 2  # Strong Buy action

    def plot_confusion_matrix(self):
        if self.actions_df is None:
            raise ValueError("You must call evaluate_last_3_days() before plotting the confusion matrix.")

        y_true = self.actions_df['Classification'].values
        y_pred = self.actions_df['Action'].values

        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Sell (-1)", "Hold (0)", "Buy (1)", "Strong Buy (2)"])

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title("Confusion Matrix: Classification vs Action")
        plt.tight_layout()
        plt.savefig("confusion_matrix_aapl.png", dpi=300)
        print("Confusion matrix saved to plots/confusion_matrix_aapl.png.")
        plt.close()


if __name__ == '__main__':
    processing = ProcessingData("4_data_w_indicators/AAPL_tech_ind.csv")
    processing.calculate_price_differences()
    q1, q2 = processing.calculate_quantiles()
    processing.classify_differences(q1, q2)
    processing.evaluate_last_3_days()
    processing.plot_confusion_matrix()

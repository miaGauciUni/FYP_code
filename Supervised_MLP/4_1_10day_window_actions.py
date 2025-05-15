import pandas as pd
import numpy as np
import os

class ProcessingData:
    def __init__(self, file_path, stock_name, output_folder):
        self.df = pd.read_csv(file_path)
        self.stock_name = stock_name
        self.output_folder = output_folder
        print(f"Loaded data for {stock_name}")

    def calculate_price_differences(self):
        self.df['Price_Difference'] = self.df['Closing'].diff()
        print("Price differences calculated.")

    def classify_differences(self):
        price_differences = self.df['Price_Difference'].dropna()
        q1 = np.quantile(price_differences[price_differences > 0], 0.45)
        q2 = np.quantile(price_differences[price_differences > 0], 0.75)

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
        print("Price movements classified.")

    def generate_windowed_data(self, window=15):
        features = ['Closing', 'EMA_5', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
        windowed_rows = []

        for i in range(window, len(self.df)):
            row = []
            for feat in features:
                values = self.df[feat].iloc[i - window:i].values  # oldest to most recent
                for j in range(window):  # from _10 to _1
                    row.append(values[j])
            action = self.df['Classification'].iloc[i]
            row.append(action)
            windowed_rows.append(row)

        # Generate proper column names
        cols = []
        for feat in features:
            for j in range(window, 0, -1):
                cols.append(f"{feat}_{j}")
        cols.append("Action")

        df_out = pd.DataFrame(windowed_rows, columns=cols)
        output_path = os.path.join(self.output_folder, f"{self.stock_name}_15day_tech_action.csv")
        df_out.to_csv(output_path, index=False)
        print(f"Saved 15-day windowed data to {output_path}")

def process_all_stocks(stock_list, input_folder, output_folder):
    for stock in stock_list:
        input_path = os.path.join(input_folder, f"{stock}_tech_ind.csv")
        processor = ProcessingData(input_path, stock, output_folder)
        processor.calculate_price_differences()
        processor.classify_differences()
        processor.generate_windowed_data(window=15)

if __name__ == '__main__':
    stock_list = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NVDA"]
    process_all_stocks(
        stock_list,
        input_folder="4_data_w_indicators",
        output_folder="5_processed_with_actions"
    )

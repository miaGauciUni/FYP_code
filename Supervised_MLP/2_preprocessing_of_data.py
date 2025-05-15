import pandas as pd
import matplotlib.pyplot as plt
import os


# cleaning and checking for any outliers
class PreprocessingOfData:
    def __init__(self, file_name):
        """
        Initializes the class with the CSV file name and loads the data into a DataFrame.
        """
        self.file_name = file_name
        self.df = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def clean_data(self):

        initial_rows = self.df.shape[0]

        # Drop rows with NaN values
        self.df.dropna(inplace=True)

        # Report how many rows were removed
        removed_rows = initial_rows - self.df.shape[0]
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows containing NaN values.")
        else:
            print("No NaN values found in the dataset.")

    def save_clean_data(self, output_file_name):

        self.df.to_csv(output_file_name, index=False)
        print(f"Cleaned data saved as {output_file_name}.")

    def plot_stock_data(self):

        # Ensure the 'Date' column is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Line plot of the 'Closing' price against 'Date'
        plt.figure(figsize=(10, 6))
        plt.plot(self.df['Date'], self.df['Closing'], label='Closing Price', color='blue')

        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.title(f'{ticker} Stock Price Over Time ')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()

    def detect_outliers_modified_zscore(self):

        median = self.df['Closing'].median()
        mad = (self.df['Closing'] - median).abs().median()

        # Calculate Modified Z-scores
        modified_z_scores = 0.6745 * (self.df['Closing'] - median) / mad

        # Define a threshold for identifying outliers
        threshold = 3.5  # Commonly used threshold
        outliers = self.df[modified_z_scores > threshold]

        print(f"Found {outliers.shape[0]} outliers using Modified Z-score method.")
        return outliers

    def plot_outliers(self):

        # Ensure the 'Date' column is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Detect outliers using the modified z-score method
        outliers = self.detect_outliers_modified_zscore()

        # Line plot of the 'Closing' price against 'Date'
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Date'], self.df['Closing'], label='Closing Price', color='blue', s=5)

        # Highlight the outliers as red dots
        plt.scatter(outliers['Date'], outliers['Closing'], label='Outliers', color='red', s=40)

        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.title(f'{ticker} Stock Price with Outliers Highlighted ')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == '__main__':
    # Define the list of ticker symbols
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NVDA"]

    # Iterate through each ticker and process the corresponding raw data file
    for ticker in tickers:
        # Construct the file name for the raw data
        file_name = f"1_raw_data/{ticker}_raw.csv"

        if not os.path.exists(file_name):
            print(f"File {file_name} not found, skipping.")
            continue

        # Initialize the preprocessing class with the raw CSV file
        preprocessing = PreprocessingOfData(file_name)

        # Clean the data by removing rows with NaN values
        preprocessing.clean_data()

        # Save the cleaned data to a new file
        output_file_name = f"3_clean_raw_data/{ticker}_clean.csv"
        preprocessing.save_clean_data(output_file_name)

        # Plot the stock data (line plot)
        preprocessing.plot_stock_data()

        # Plot the stock data with outliers highlighted
        preprocessing.plot_outliers()

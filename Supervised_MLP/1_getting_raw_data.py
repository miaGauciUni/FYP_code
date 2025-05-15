import yfinance as yf
import pandas as pd


def download_and_save_stock_data():
    # Define the list of ticker symbols
    tickers = ["AAPL","MSFT","GOOGL","META","AMZN", "TSLA","NVDA"]

    # Define the time periods
    start_raw = '2020-01-01'
    end_raw = '2023-01-01'  # End date is inclusive for raw data
    start_validation = '2023-01-01'
    end_validation = '2024-03-01'  # End date is inclusive for validation data

    for ticker in tickers:
        # Download historical data for raw data period
        raw_df = yf.download(ticker, start=start_raw, end=end_raw, interval='1d')

        if raw_df.empty:
            print(f"No data found for {ticker} in raw data period.")
            continue  # Skip this ticker if no data is found

        # Keep only the 'Date', 'Close', and 'Volume' columns for raw data
        raw_df = raw_df[['Close', 'Volume']]
        raw_df.reset_index(inplace=True)
        raw_df.columns = ['Date', 'Closing', 'Volume']

        # Save the raw DataFrame to a CSV file
        raw_csv_filename = f"1_raw_data/{ticker}_raw.csv"
        raw_df.to_csv(raw_csv_filename, index=False)
        print(f"CSV file saved for {ticker} as {raw_csv_filename}")

        # Download historical data for validation data period
        validation_df = yf.download(ticker, start=start_validation, end=end_validation, interval='1d')

        if validation_df.empty:
            print(f"No data found for {ticker} in validation data period.")
            continue  # Skip this ticker if no data is found

        # Keep only the 'Date' and 'Close' columns for validation data
        validation_df = validation_df[['Close']]
        validation_df.reset_index(inplace=True)
        validation_df.columns = ['Date', 'Closing']  # Rename 'Close' to 'Closing'

        # Save the validation DataFrame to a CSV file
        validation_csv_filename = f"2_validation_data/{ticker}_validation.csv"
        validation_df.to_csv(validation_csv_filename, index=False)
        print(f"CSV file saved for {ticker} as {validation_csv_filename}")


def load_csvs_to_list(tickers):
    data_frames = []

    for ticker in tickers:
        # Define the CSV file name based on the ticker symbol for raw data
        raw_csv_filename = f"1_raw_data/{ticker}_raw.csv"

        # Define the CSV file name based on the ticker symbol for validation data
        validation_csv_filename = f"2_validation_data/{ticker}_validation.csv"

        try:
            # Load the raw CSV file into a DataFrame
            raw_df = pd.read_csv(raw_csv_filename)
            data_frames.append(raw_df)
            print(f"Loaded {raw_csv_filename} into the list.")
        except FileNotFoundError:
            print(f"CSV file for {ticker} (raw data) not found!")

        try:
            # Load the validation CSV file into a DataFrame
            validation_df = pd.read_csv(validation_csv_filename)
            data_frames.append(validation_df)
            print(f"Loaded {validation_csv_filename} into the list.")
        except FileNotFoundError:
            print(f"CSV file for {ticker} (validation data) not found!")

    return data_frames


if __name__ == '__main__':
    # Download data and save it to CSV files
    download_and_save_stock_data()

    # Define the list of ticker symbols for which CSV files have been saved
    tickers = ["AAPL","MSFT","GOOGL","META","AMZN","TSLA","NVDA"]

    # Load all the CSV files into a list of DataFrames
    stock_data_list = load_csvs_to_list(tickers)

    # Now stock_data_list contains all the data in list form
    if stock_data_list:
        # For demonstration, print the first few rows of the first stock's raw data
        print(stock_data_list[0].head())  # Print first few rows of AAPL raw data

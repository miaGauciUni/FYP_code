import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

class AgentEnvironment:
    def __init__(self, initial_cash, stock_data_files, model_files):
        # Initial cash and total portfolio value
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.total_value = initial_cash

        # Portfolio & price bookkeeping
        self.portfolio = {s: 0 for s in stock_data_files}
        self.current_prices = {}
        self.predictions = {}

        # Load data + models
        self.validation_data = {s: pd.read_csv(p) for s,p in stock_data_files.items()}
        self.models = {s: pickle.load(open(m, 'rb')) for s,m in model_files.items()}

        # Fixed feature list (3-day window hardcoded here)
        self.feature_columns = [
            'Closing_15', 'Closing_14', 'Closing_13', 'Closing_12', 'Closing_11',
                'Closing_10', 'Closing_9', 'Closing_8', 'Closing_7', 'Closing_6',
                'Closing_5', 'Closing_4', 'Closing_3', 'Closing_2', 'Closing_1',

                'EMA_5_15', 'EMA_5_14', 'EMA_5_13', 'EMA_5_12', 'EMA_5_11',
                'EMA_5_10', 'EMA_5_9', 'EMA_5_8', 'EMA_5_7', 'EMA_5_6',
                'EMA_5_5', 'EMA_5_4', 'EMA_5_3', 'EMA_5_2', 'EMA_5_1',

                'RSI_15', 'RSI_14', 'RSI_13', 'RSI_12', 'RSI_11',
                'RSI_10', 'RSI_9', 'RSI_8', 'RSI_7', 'RSI_6',
                'RSI_5', 'RSI_4', 'RSI_3', 'RSI_2', 'RSI_1',

                'MACD_15', 'MACD_14', 'MACD_13', 'MACD_12', 'MACD_11',
                'MACD_10', 'MACD_9', 'MACD_8', 'MACD_7', 'MACD_6',
                'MACD_5', 'MACD_4', 'MACD_3', 'MACD_2', 'MACD_1',

                'BB_upper_15', 'BB_upper_14', 'BB_upper_13', 'BB_upper_12', 'BB_upper_11',
                'BB_upper_10', 'BB_upper_9', 'BB_upper_8', 'BB_upper_7', 'BB_upper_6',
                'BB_upper_5', 'BB_upper_4', 'BB_upper_3', 'BB_upper_2', 'BB_upper_1',

                'BB_lower_15', 'BB_lower_14', 'BB_lower_13', 'BB_lower_12', 'BB_lower_11',
                'BB_lower_10', 'BB_lower_9', 'BB_lower_8', 'BB_lower_7', 'BB_lower_6',
                'BB_lower_5', 'BB_lower_4', 'BB_lower_3', 'BB_lower_2', 'BB_lower_1'
        ]

        # Storage
        self.portfolio_values = []
        self.monthly_returns = []
        self.daily_actions = []

    def update_stock_prices(self, day):
        for s, df in self.validation_data.items():
            # assume first column is closing price
            self.current_prices[s] = df.iloc[day, 0]

    def make_decision(self, day):
        """
        Use models to make decisions based on validation data for the given day.
        """
        self.predictions.clear()
        for stock, model in self.models.items():
            df = self.validation_data[stock]
            # slice out a 1×N DF with the original feature names
            feat_df = df.loc[[day], self.feature_columns]
            self.predictions[stock] = model.predict(feat_df)[0]

    def calculate_ratios(self):
        pos = {s:p for s,p in self.predictions.items() if p>0}
        total_w = sum(2 if p==2 else 1 for p in pos.values())
        ratios = {s:0 for s in self.portfolio}
        if total_w>0:
            for s,p in pos.items():
                w = 2 if p==2 else 1
                ratios[s] = w/total_w
        return ratios

    def log_action(self, stock, action, shares, price):
        self.daily_actions.append(f"{action.capitalize()} {shares} of {stock} at ${price:.2f}")

    def allocate_funds(self):
        # first‐day equal split?
        if all(v==0 for v in self.portfolio.values()):
            inv = self.cash / len(self.portfolio)
            for s in self.portfolio:
                n = inv // self.current_prices[s]
                if n>0:
                    self.portfolio[s]+=n
                    self.cash -= n*self.current_prices[s]
                    self.log_action(s,'buy',n,self.current_prices[s])
            return

        ratios = self.calculate_ratios()
        for s,p in self.predictions.items():
            if p==-1 and self.portfolio[s]>0:
                self.cash += self.portfolio[s]*self.current_prices[s]
                self.log_action(s,'sell',self.portfolio[s],self.current_prices[s])
                self.portfolio[s]=0
            elif p>0:
                alloc = self.cash * ratios[s]
                n = alloc // self.current_prices[s]
                if n>0:
                    self.portfolio[s]+=n
                    self.cash -= n*self.current_prices[s]
                    self.log_action(s,'buy',n,self.current_prices[s])

    def get_portfolio_value(self):
        pv = sum(self.portfolio[s]*self.current_prices[s] for s in self.portfolio)
        self.total_value = self.cash + pv
        return self.total_value

    def run_simulation(self):
        # use the shortest ticker series
        num_days = min(len(df) for df in self.validation_data.values())
        days_per_month = 21

        start_val = self.total_value
        month = 1

        for day in range(num_days):
            self.daily_actions.clear()
            self.update_stock_prices(day)
            self.make_decision(day)
            self.allocate_funds()

            tv = self.get_portfolio_value()
            self.portfolio_values.append(tv)

            # end-of-month
            if (day+1)%days_per_month==0 or day==num_days-1:
                mr = (tv - start_val)/start_val*100
                self.monthly_returns.append(mr)
                print(f"End of Month {month}: Return = {mr:.2f}%")
                start_val = tv
                month+=1

        print("Avg Monthly Return:", np.mean(self.monthly_returns))

    def buy_and_hold(self):
        # number of trading days in the shortest series
        num_days = min(len(df) for df in self.validation_data.values())

        # split ALL your cash equally
        inv_per_stock = self.initial_cash  / len(self.portfolio)

        # decide how many shares you buy on day 0 (use iloc[0,0] for the closing price)
        shares = {}
        for s, df in self.validation_data.items():
            price0 = df.iloc[0, 0]  # <-- first column, first row
            shares[s] = inv_per_stock / price0

        # now track the portfolio value each day
        hold_values = []
        for day in range(num_days):
            # update prices exactly like in your main loop
            self.update_stock_prices(day)
            total = sum(shares[s] * self.current_prices[s] for s in shares)
            hold_values.append(total)

        print(f"\nBuy & Hold Final Value: ${hold_values[-1]:.2f}")
        return hold_values


if __name__=='__main__':
    initial_cash = 10000
    stock_data_files = {
        'AAPL':'2_validation_data/AAPL_processed_validation_20day.csv',
        'AMZN':'2_validation_data/AMZN_processed_validation_20day.csv',
        'META':'2_validation_data/META_processed_validation_20day.csv',
        'GOOGL':'2_validation_data/GOOGL_processed_validation_20day.csv',
        'MSFT':'2_validation_data/MSFT_processed_validation_20day.csv',
        'NVDA':'2_validation_data/NVDA_processed_validation_20day.csv',
        'TSLA':'2_validation_data/TSLA_processed_validation_20day.csv',
    }
    model_files = {
        s:f"6_stock_models/{s}_MLP_model_15day.pkl"
        for s in stock_data_files
    }

    import pickle

    # Run simulation
    env = AgentEnvironment(initial_cash, stock_data_files, model_files)
    env.run_simulation()

    # Get buy-and-hold data
    buy_and_hold_values = env.buy_and_hold()

    # Save both to pickle files for future plotting
    with open("rolling_window_experiment/20day_model_values.pkl", "wb") as f:
        pickle.dump(env.portfolio_values, f)

    with open("0_Buy_and_Hold/20day_buy_hold_values.pkl", "wb") as f:
        pickle.dump(buy_and_hold_values, f)

    # Optional: also save as CSV for inspection or cross-compatibility
    pd.DataFrame({
        "Model": env.portfolio_values,
        "Buy&Hold": buy_and_hold_values
    }).to_csv("rolling_window_experiment/20day_portfolio_comparison.csv", index=False)

    # Plot
    plt.figure(figsize=(8, 4))
    days = range(len(env.portfolio_values))
    plt.plot(days, env.portfolio_values, label='Model')
    plt.plot(days, buy_and_hold_values, label='Buy & Hold', linestyle='--')
    plt.legend()
    plt.title("Portfolio Over Time")
    plt.show()


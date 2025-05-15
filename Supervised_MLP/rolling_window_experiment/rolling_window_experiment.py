import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Load pickled results
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Load model portfolios
model_3 = load_pickle("3day_model_values.pkl")
model_5 = load_pickle("5day_model_values.pkl")
model_10 = load_pickle("10day_model_values.pkl")
model_15 = load_pickle("20day_model_values.pkl")
buy_hold_15 = load_pickle("../0_Buy_and_Hold/20day_buy_hold_values.pkl")

# Load real dates from a 15-day CSV (or raw if consistent)
df = pd.read_csv("../2_validation_data/AAPL_validation.csv")
dates = df["Date"].values[:len(model_15)]  # Truncate to match min length

# Truncate all series
min_len = len(dates)
model_3 = model_3[:min_len]
model_5 = model_5[:min_len]
model_10 = model_10[:min_len]
model_15 = model_15[:min_len]
buy_hold_15 = buy_hold_15[:min_len]

# Create plot
plt.figure(figsize=(12,6))
plt.plot(dates, model_3, label="3-Day Window")
plt.plot(dates, model_5, label="5-Day Window")
plt.plot(dates, model_10, label="10-Day Window")
plt.plot(dates, model_15, label="15-Day Window")
plt.plot(dates, buy_hold_15, label="Buy & Hold", linestyle="--")

# Set readable x-axis
step = len(dates) // 10  # Show ~10 x-axis ticks
plt.xticks(ticks=range(0, len(dates), step), labels=dates[::step], rotation=45)

plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Model Performance by Window Size vs Buy & Hold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("all_windows_vs_buyhold_with_dates_cleaned.png", dpi=600)
plt.show()

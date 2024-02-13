import yfinance as yf
import pandas as pd
import sqlite3
from datetime import date
import numpy as np

ticker = 'INFY'
stock_name = "INFOSYS"
end_date = date.today().strftime('%Y-%m-%d')  # Get today's date in the correct format
stock_data = yf.download(ticker, start='2023-07-01', end=end_date)

# Keep only 'Adj Close' and 'Volume' columns
stock_data = stock_data[['Adj Close', 'Volume']]

# Rename 'Adj Close' to 'Close'
stock_data = stock_data.rename(columns={'Adj Close': 'Close'})

# Set the index name to 'Date'
stock_data = stock_data.rename_axis('Date')

# Save data to SQLite database
db_filename = "stock_data.db"
conn = sqlite3.connect(db_filename)
stock_data.to_sql(stock_name, conn, if_exists='replace', index=True)
conn.close()

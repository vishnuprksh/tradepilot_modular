import yfinance as yf
import pandas as pd
import sqlite3
from datetime import date
import numpy as np

ticker = 'RELIANCE.NS'
stock_data = yf.download(ticker, start='2023-07-01', end='2023-12-31')

# Keep only 'Adj Close' and 'Volume' columns
stock_data = stock_data[['Adj Close', 'Volume']]
stock_data = stock_data.rename_axis('Date')


# Save data to SQLite database
db_filename = "stock_data.db"
conn = sqlite3.connect(db_filename)
stock_data.to_sql(ticker, conn, if_exists='replace', index=True)
conn.close()

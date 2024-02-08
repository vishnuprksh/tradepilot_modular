import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from functions import load_data_from_database, preprocess_data, fill_missing_dates, predict_tomorrow_price, fetch_tweet_data, merge_data
from functions import pre_processing_and_prediction

# Streamlit App
st.title("Stock Price Prediction App")

# Dropdown to select the stock ticker
ticker = st.selectbox("Select Ticker", ["RELIANCE.NS", "TEST"])

# Specify the desired date range
start_date = '2023-07-01'
end_date = '2023-12-31'

# Load data and train model
stock_data = load_data_from_database(ticker)

# Fill missing dates with zeros
filled_stock_data = fill_missing_dates(stock_data, start_date, end_date)

# get twitter data
tweet_data = fetch_tweet_data()

# Preprocess the data
sentiment_df = preprocess_data(tweet_data)

merged_df = merge_data(filled_stock_data, sentiment_df)

merged_df['Target'] = merged_df['Close'].shift(-1)

final_df = merged_df[:-1]

final_df.to_csv('final_df.csv', index=False)

accuracy, next_day_price = pre_processing_and_prediction(final_df)


# Display historical stock prices chart
st.subheader("Stock Price History (Last 6 Months)")
st.line_chart(stock_data.set_index('Date')['Close'])

# Display Mean Squared Error of the model
st.write(f"R2 Score of the Model: {accuracy: .2f}")


# Display predicted stock price for tomorrow
st.subheader("Predicted Stock Price for Tomorrow")
st.write(f"The predicted stock price for tomorrow is: {next_day_price:.2f}")


import streamlit as st
import yfinance as yf
import os
from dotenv import load_dotenv
import requests
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from functions import load_data_from_database, preprocess_data, fill_missing_dates, predict_tomorrow_price, fetch_tweet_data, merge_data
from functions import pre_processing_and_prediction
load_dotenv()

# Centered Image
centered_image_html = """
<div style="display: flex; justify-content: center; align-items: center;">
    <h1>Trade Pilot</h1>
</div>
"""

st.markdown(centered_image_html, unsafe_allow_html=True)

# 2. Search Bar
st.text_input("Enter Company or Ticker", placeholder="AAPL, TSLA, etc.", key="search_input")

if not st.session_state.search_input:
    # Function to get real-time market data
    def get_market_data(ticker_symbol):
        # Fetch the stock data
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history()

        # Get the last closing price
        last_price = data['Close'].iloc[-1]

        # Calculate the change percentage
        change_percent = (last_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100

        return last_price, change_percent

    # Replace the following symbols with the ones you are interested in (e.g., '^DJI', '^GSPC', '^IXIC')
    ticker_1 = '^NSEI'
    ticker_2 = '^BSESN'
    ticker_3 = '^NSEBANK'

    price_1, percentage_1 = get_market_data(ticker_1)
    price_2, percentage_2 = get_market_data(ticker_2)
    price_3, percentage_3 = get_market_data(ticker_3)

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY 50", f"{price_1:.2f}", f"{percentage_1:.2f}%")
    col2.metric("SENSEX", f"{price_2:.2f}", f"{percentage_2:.2f}%")
    col3.metric("NIFTY BANK", f"{price_3:.2f}", f"{percentage_3:.2f}%")

    def fetch_news(api_key):
        url = f'https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey={api_key}'
        response = requests.get(url)
        return response.json()

    def display_news(news_data):
        if news_data.get('status') == 'ok':
            articles = news_data.get('articles', [])
            for i, article in enumerate(articles[:10], start=1):
                st.write(f"**{i}. [{article['title']}]({article['url']})**")

                # Check if 'urlToImage' is provided in the response
                if 'urlToImage' in article and article['urlToImage']:
                    try:
                        st.image(article['urlToImage'], caption=article['description'], use_column_width=True)
                    except Exception as e:
                        st.warning(f"Failed to display image for article {i}: {e}")
                else:
                    st.warning(f"No image available for article {i}")

                st.write(f"Source: {article['source']['name']}")
                st.write("------")

    api_key = os.getenv("NEWS_API")  # Replace with your News API key
    st.title("News Headlines")
    news_data = fetch_news(api_key)

    if not news_data:
        st.error("Failed to fetch news. Check your API key and try again.")
    else:
        display_news(news_data)
else:
    st.write(st.session_state.search_input)
    # Specify the desired date range
    start_date = '2023-07-01'
    end_date = '2023-12-31'

    # Load data and train model
    stock_data = load_data_from_database(st.session_state.search_input)

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

    accuracy, next_day_price, fig = pre_processing_and_prediction(final_df)


    # Display historical stock prices chart
    st.subheader("Stock Price History (Last 6 Months)")
    st.plotly_chart(fig)

    # Display Mean Squared Error of the model
    st.write(f"R2 Score of the Model: {accuracy: .2f}")


    # Display predicted stock price for tomorrow
    st.subheader("Predicted Stock Price for Tomorrow")
    st.write(f"The predicted stock price for tomorrow is: {next_day_price:.2f}")    

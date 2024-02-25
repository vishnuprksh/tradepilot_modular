import streamlit as st
import yfinance as yf
import os
import requests
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import toml
from functions import load_stock_data_from_db, preprocess_data, fill_missing_dates, predict_tomorrow_price, fetch_tweet_data, merge_data
from functions import pre_processing_and_prediction, get_stock_details_from_gemini, get_market_data, update_stock_date
config_data = toml.load("config.toml")

# Centered Image
centered_header = """
<div style="display: flex; justify-content: center; align-items: center;">
    <h1 style='color: #F7E987;'>Trade Pilot</h1>
</div>
"""

# Centered Image
centered_tag = """
<div style="display: flex; justify-content: center; align-items: center;">
    <h5 style='color: #5B9A8B'>Unleash the Power of Social Media for Your Stock Investments!</h5>
</div>
"""


st.markdown(centered_header, unsafe_allow_html=True)
st.markdown(centered_tag, unsafe_allow_html=True)


# Replace the following symbols with the ones you are interested in (e.g., '^DJI', '^GSPC', '^IXIC')
ticker_1 = '^NSEI'
ticker_2 = '^BSESN'
ticker_3 = '^NSEBANK'

price_1, percentage_1 = get_market_data(ticker_1)
price_2, percentage_2 = get_market_data(ticker_2)
price_3, percentage_3 = get_market_data(ticker_3)

# Display metrics in columns
col1, col2, col3 = st.columns(3)

# Display metrics with reduced font size for prices
col1.metric("NIFTY 50", f"{price_1:.2f}", f"{percentage_1:.2f}%")
col2.metric("SENSEX", f"{price_2:.2f}", f"{percentage_2:.2f}%")
col3.metric("NIFTY BANK", f"{price_3:.2f}", f"{percentage_3:.2f}%")

# List of suggested tickers
stock_list = ["", "RELIANCE", "INFOSYS", "WEBELSOLAR", "WIPRO"]
ticker_list = ["", "RELIANCE.NS", "INFY.NS", "WEBELSOLAR.NS", "WIPRO.NS"]

# 2. Search Bar with auto-suggest
st.selectbox("Select Ticker", stock_list, key="search_input")

stock_bool = update_stock_date(stock_list, ticker_list)


if not st.session_state.search_input:
    if stock_bool:
        st.success("Stock data updated!")

    # How it Works Section
    st.subheader("❔ How Does We Work?")
    st.write("Our intelligent algorithms analyze millions of tweets to gauge the overall sentiment towards specific stocks. Whether it's bullish, bearish, or neutral, we've got you covered.")

    # Real-Time Analysis Section
    st.subheader("🌐 Real-Time Analysis:")
    st.write("Stay ahead of the curve with our real-time sentiment analysis. We continuously monitor social media trends and instantly update our predictions, ensuring you have the latest information at your fingertips.")

    # Interactive Charts Section
    st.subheader("📉 Interactive Charts:")
    st.write("Visualize sentiment trends with easy-to-read charts. Track how social media sentiment aligns with stock movements and make informed decisions based on comprehensive data.")

    # Secure and Reliable Section
    st.subheader("🔐 Secure and Reliable:")
    st.write("Rest easy knowing that your data and investments are secure. We prioritize the privacy and security of our users, providing a reliable platform you can trust.")


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

    api_key = config_data["API"]["NEWS_API_KEY"]
    st.header("Latest News Headlines")
    news_data = fetch_news(api_key)

    if not news_data:
        st.error("Failed to fetch news. Check your API key and try again.")
    else:
        display_news(news_data)
else:

    stock_details_result = get_stock_details_from_gemini(st.session_state.search_input)
    st.markdown(stock_details_result, unsafe_allow_html=True)

    # Specify the desired date range
    start_date = '2023-07-01'
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Load data and train model
    stock_data = load_stock_data_from_db(st.session_state.search_input, start_date, end_date)

    # get twitter data
    tweet_data = fetch_tweet_data(st.session_state.search_input, start_date, end_date)

    # Preprocess the data
    sentiment_df = preprocess_data(tweet_data)

    merged_df = merge_data(stock_data, sentiment_df)

    merged_df['Target'] = merged_df['Close'].shift(-1)

    final_df = merged_df[:-1]

    # final_df.to_csv('final_df.csv', index=False)

    accuracy, next_day_price, fig = pre_processing_and_prediction(final_df)


    # Display historical stock prices chart
    # st.subheader("Stock Price History")
    st.plotly_chart(fig)

    # Display Mean Squared Error of the model
    # st.write(f"R2 Score of the Model: {accuracy: .2f}")


    # Display predicted stock price for tomorrow
    st.markdown("""
    <style>
        .price-box {
            background-color: #f0f5f5; /* Light colored background */
            color: black;
            border: 1px solid #ccc;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 28px; /* Large font */
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='price-box'>The predicted stock price for tomorrow : {next_day_price:.2f}</div>", unsafe_allow_html=True) 


    st.text("")  # This creates a new line
    st.text("")  # This creates a new line
    st.text("")  # This creates a new line
    # Disclaimer Section
    st.warning("Disclaimer: Stock market predictions involve inherent risks. Our platform provides insights based on social media sentiments, but users are encouraged to conduct thorough research and consult with financial experts before making investment decisions.")


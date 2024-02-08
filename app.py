import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import sqlite3

# Function to load historical stock data and train a basic model
def load_data_from_database(ticker, database_path="stock_data.db"):
    # Connect to the SQLite database
    connection = sqlite3.connect(database_path)

    # Query the stock data from the database
    query = f"SELECT * FROM stock_data WHERE Ticker = '{ticker}' AND Date BETWEEN '2023-07-01' AND '2024-02-01'"
    stock_data = pd.read_sql_query(query, connection)

    # Close the database connection
    connection.close()

    # Feature engineering: Extract day, month, and year from Date
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Day'] = stock_data['Date'].dt.day
    stock_data['Month'] = stock_data['Date'].dt.month
    stock_data['Year'] = stock_data['Date'].dt.year

    # Prepare data for training
    X = stock_data[['Day', 'Month', 'Year']]
    y = stock_data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the stock prices for the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error as a metric
    mse = mean_squared_error(y_test, y_pred)

    return stock_data, model, mse

# Function to predict stock price for tomorrow
def predict_tomorrow_price(model, last_date):
    tomorrow_date = last_date + timedelta(days=1)
    tomorrow_data = {'Day': tomorrow_date.day, 'Month': tomorrow_date.month, 'Year': tomorrow_date.year}
    
    # Correctly reshape the input data for prediction
    tomorrow_data_df = pd.DataFrame([tomorrow_data])
    
    tomorrow_price = model.predict(tomorrow_data_df)[0]
    return tomorrow_price


# Streamlit App
st.title("Stock Price Prediction App")

# Dropdown to select the stock ticker
ticker = st.selectbox("Select Ticker", ["RELIANCE.NS", "TEST"])

# Load data and train model
stock_data, model, mse = load_data(ticker)

# Display historical stock prices chart
st.subheader("Stock Price History (Last 6 Months)")
st.line_chart(stock_data.set_index('Date')['Close'])

# Display Mean Squared Error of the model
st.write(f"Mean Squared Error of the Model: {mse}")

# Predict tomorrow's stock price
last_date = stock_data['Date'].max()
tomorrow_price = predict_tomorrow_price(model, last_date)

# Display predicted stock price for tomorrow
st.subheader("Predicted Stock Price for Tomorrow")
st.write(f"The predicted stock price for tomorrow ({last_date + timedelta(days=1)}) is: {tomorrow_price:.2f}")

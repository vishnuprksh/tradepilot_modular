import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import sqlite3
from textblob import TextBlob  
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
from sklearn.metrics import r2_score
import plotly.express as px
import tensorflow as tf



def preprocess_data(tweet_data):

    # Function to calculate sentiment score using TextBlob
    def get_sentiment_score(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    # Apply sentiment analysis and create a new column 'sentiment_score'
    tweet_data['Sentiment_score'] = tweet_data['text'].apply(get_sentiment_score)

    # Group by formatted date and calculate the mean sentiment score
    grouped_data = tweet_data.groupby('Date').agg({
        'Sentiment_score': 'mean',
        'No_of_likes': 'sum',
        'No_of_comments': 'sum',
        'text': 'count'  # Add a new column for sentiment count
    }).reset_index()

    # Rename the 'text' column to 'count_of_tweets'
    grouped_data.rename(columns={'text': 'Count_of_tweets'}, inplace=True)

    return grouped_data





def generate_date_range(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    return pd.DataFrame({'Date': date_range})




def fetch_tweet_data(database_path="tweets_database.db"):
    # Connect to the SQLite database
    connection = sqlite3.connect(database_path)

    # Query the tweet_data from the database
    query = "SELECT * FROM tweet_data"
    tweet_data = pd.read_sql_query(query, connection)

    # Close the database connection
    connection.close()

    tweet_data.to_csv('tweet_data.csv', index=False)   

    return tweet_data





def fill_missing_dates(stock_data, start_date, end_date):
    date_range_df = generate_date_range(start_date, end_date)
    
    filled_stock_data = pd.merge(date_range_df, stock_data, on='Date', how='left').fillna(0)
    filled_stock_data.to_csv('filled_stock_data.csv', index=False)

    return filled_stock_data






# Function to load historical stock data and train a basic model
def load_stock_data_from_db(ticker, start_date='2023-07-01', end_date='2023-12-31', database_path="stock_data.db"):
    # Connect to the SQLite database
    connection = sqlite3.connect(database_path)

    # Query the stock data from the database with specified start_date and end_date
    query = f"SELECT * FROM reliance WHERE Date BETWEEN '{start_date}' AND '{end_date}'"
    stock_data = pd.read_sql_query(query, connection)

    # Close the database connection
    connection.close()

    # Feature engineering: Extract day, month, and year from Date
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    stock_data.to_csv(f'{ticker}_stock_data.csv', index=False)    

    return stock_data




def merge_data(stock_data, sentiment_df):
    # Convert the 'formatted_date' column in df1 to datetime64[ns]
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

    # Convert the 'Date' column in df2 to datetime64[ns]
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Merge dataframes on the 'formatted_date' and 'Date' columns
    merged_df = pd.merge(sentiment_df, stock_data, left_on='Date', right_on='Date', how='inner')
    merged_df = merged_df[merged_df['Close'] != 0]

    merged_df.to_csv('merged_final.csv', index=False)

    return merged_df


def pre_processing_and_prediction(final_df):
    # input_data = final_df[["Close", 'Volume']].values
    input_data = final_df[["Close", 'Sentiment_score', 'No_of_likes', 'No_of_comments', 'Count_of_tweets', 'Volume']].values


    sc = MinMaxScaler()
    input_scaled = sc.fit_transform(input_data)

    X = []
    y = []

    for i in range(1, len(input_scaled)):
        X.append(input_scaled[i - 1:i, :])  # Include all features for X
        y.append(input_scaled[i, 0])  # Use only 'TARGET' as the target

    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    # x = keras.layers.LSTM(140, return_sequences=True)(inputs)
    # x = keras.layers.Dropout(0.3)(x)
    # x = keras.layers.LSTM(140, return_sequences=True)(x)
    # x = keras.layers.Dropout(0.3)(x)
    # x = keras.layers.LSTM(140)(x)
    # outputs = keras.layers.Dense(1, activation='linear')(x)

    # model = keras.Model(inputs=inputs, outputs=outputs)
    # model.compile(optimizer='adam', loss='mse')
    # model.summary()

    # history = model.fit(
    #     X_train, y_train,
    #     epochs=50,
    #     batch_size=32,
    #     validation_split=0.2
    # )

    # model.save("reliance_model.keras")
    model = tf.keras.models.load_model("reliance_model.keras")

    # Make predictions for the next day
    next_day_features = input_scaled[-1].reshape((1, X_test.shape[1], X_test.shape[2]))
    next_day_prediction_scaled = model.predict(next_day_features)[0][0]
    # Inverse transform to get the prediction in the original scale
    # scaled_prediction = np.array([[next_day_prediction_scaled, 0]])  # Add placeholders for other features
    scaled_prediction = np.array([[next_day_prediction_scaled, 0, 0, 0, 0, 0]])  # Add placeholders for other features

    true_next_day_price = sc.inverse_transform(scaled_prediction)[0, 0]

    # Calculate accuracy
    accuracy = r2_score(y_test, model.predict(X_test))


    def draw_graph():
        predicted = model.predict(X)

        df_predicted = pd.DataFrame({'Date': final_df['Date'][1:], 'predictions': predicted[:, 0], 'Close': input_scaled[1:, 0]})
        # df_predicted['difference'] = df_predicted['Close'] - df_predicted['predictions']
        fig = px.line(df_predicted, x='Date', y=df_predicted.columns[1:], title='Original vs Prediction')
        return fig

    fig = draw_graph()

    return accuracy, true_next_day_price, fig



# Function to predict stock price for tomorrow
def predict_tomorrow_price(model, last_date):
    tomorrow_date = last_date + timedelta(days=1)
    tomorrow_data = {'Day': tomorrow_date.day, 'Month': tomorrow_date.month, 'Year': tomorrow_date.year}
    
    # Correctly reshape the input data for prediction
    tomorrow_data_df = pd.DataFrame([tomorrow_data])
    
    tomorrow_price = model.predict(tomorrow_data_df)[0]
    return tomorrow_price

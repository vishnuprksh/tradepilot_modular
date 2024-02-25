import sqlite3
import pandas as pd

def fetch_tweet_data(database_path, table_name, start_date, end_date):
    # Connect to the SQLite database
    connection = sqlite3.connect(database_path)

    # Formulate the SQL query with specified table and date range
    query = f"SELECT * FROM {table_name} WHERE date BETWEEN '{start_date}' AND '{end_date}'"
    
    # Query the tweet_data from the database
    tweet_data = pd.read_sql_query(query, connection)

    # Close the database connection
    connection.close()

    tweet_data.to_csv('tweet_data_test.csv', index=False)

    return tweet_data

# Example usage:
database_path = "tweets_database.db"
table_name = "RELIANCE"
start_date = "2023-07-01"
end_date = "2023-07-31"

result = fetch_tweet_data(database_path, table_name, start_date, end_date)

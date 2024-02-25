import pandas as pd
from glob import glob
from datetime import datetime
import os
import sqlite3

def read_tweet_datasets(path='tweets/tweet_data_from_2023*.csv'):
    # Get the current working directory
    current_directory = os.getcwd()
    
    # Construct the full path by joining the current directory with the provided path
    full_path = os.path.join(current_directory, path)
    
    # Use glob to get a list of all matching file names
    file_list = glob(full_path)
    
    # Initialize an empty DataFrame to store the concatenated data
    all_tweets = pd.DataFrame()

    # Loop through each file and concatenate the data
    for file in file_list:
        # Read the CSV file into a DataFrame
        tweet_data = pd.read_csv(file)
        
        # Convert the 'date' column to datetime and then format it
        tweet_data['date'] = pd.to_datetime(tweet_data['date'], format="%b %d, %Y · %I:%M %p UTC").dt.strftime("%Y-%m-%d")
        
        # Concatenate the current DataFrame with the overall DataFrame
        all_tweets = pd.concat([all_tweets, tweet_data], ignore_index=True)

    return all_tweets

# Call the function
df_data = read_tweet_datasets()

# Save the DataFrame to an SQLite database
db_path = 'tweets_database.db'
table_name = 'tweet_data'

# Establish a connection to the SQLite database
conn = sqlite3.connect(db_path)

# Save the DataFrame to the database
df_data.to_sql(table_name, conn, index=False, if_exists='replace')

# Close the database connection
conn.close()

print(f'Data saved to {db_path}, Table: {table_name}')

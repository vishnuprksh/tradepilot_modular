import pandas as pd
from ntscraper import Nitter
from datetime import datetime, timedelta
import os
import sqlite3

def create_db_table_if_not_exists(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tweet_data (
            link TEXT,
            text TEXT,
            date TEXT,
            No_of_Likes INTEGER,
            No_of_tweets INTEGER
        )
    ''')
    conn.commit()

def insert_data_into_db(conn, data_list):
    cursor = conn.cursor()
    for df in data_list:
        df.to_sql('tweet_data', conn, if_exists='append', index=False)
    conn.commit()

def get_tweets(name, mode, since_date, until_date,scraper, conn):
    data_list = []  # List to store data for each day
    all_dates_skipped = True  # Set to True initially

    # Iterate over each day in the date range
    current_date = pd.to_datetime(since_date)
    end_date = pd.to_datetime(until_date)

    while current_date <= end_date:
        current_date_str = current_date.strftime("%Y-%m-%d")
        next_date = current_date + pd.Timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%d")

        # Check if data for the current date already exists in the database
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM tweet_data WHERE date = ?', (current_date_str,))
        count = cursor.fetchone()[0]

        if count > 0:
            # Data for the current date already exists, skip this date
            print(f"Data for the date {current_date_str} already exists in the database. Skipping...")
            current_date += pd.Timedelta(days=1)
            continue
        else:
            print(f"Data for the date {current_date_str} does not exist")
            try:
                all_dates_skipped = False  # Set to False if data is not available in the db even for a single date

                # Retrieve tweets for the current day
                tweets = scraper.get_tweets(name, mode=mode, since=current_date_str, until=next_date_str)

                final_tweets = []

                # Process tweets and extract relevant information
                for tweet in tweets['tweets']:
                    # Convert the 'date' field to the desired format
                    date_string = tweet['date']
                    parsed_date = datetime.strptime(date_string, "%b %d, %Y · %I:%M %p UTC")
                    formatted_date = parsed_date.strftime("%Y-%m-%d")
                    
                    # Append the data to final_tweets
                    data = [tweet['link'], tweet['text'], formatted_date, tweet['stats']['likes'], tweet['stats']['comments']]
                    final_tweets.append(data)

                # Create a DataFrame for the current day
                current_day_data = pd.DataFrame(final_tweets, columns=['link', 'text', 'date', 'No_of_Likes', 'No_of_tweets'])

                if not current_day_data.empty:
                    print(f"Tweets for the date {current_date_str} collected")
                    # Append the DataFrame to the list
                    data_list.append(current_day_data)

            except Exception as e:
                print(f"Error collecting tweets for the date {current_date_str}: {str(e)}")
                # Recreate the Nitter instance in case of an error
                scraper = Nitter(log_level=1, skip_instance_check=False)
                current_date = pd.to_datetime(since_date)
                continue

        # Move to the next day
        current_date += pd.Timedelta(days=1)

    # Insert data into SQLite database
    insert_data_into_db(conn, data_list)

    return all_dates_skipped


# Instantiate Nitter and SQLite connection outside the function
scraper = Nitter(log_level=1, skip_instance_check=False)
db_filename = 'tweet_data.db'
conn = sqlite3.connect(db_filename)

# Create the table if it doesn't exist
create_db_table_if_not_exists(conn)

# Keep calling the function until all dates are skipped
all_dates_skipped = False
while not all_dates_skipped:
    all_dates_skipped = get_tweets("reliance", 'hashtag', "2023-07-01", "2023-07-05", scraper, conn)

# Close the SQLite connection after all dates are collected
conn.close()

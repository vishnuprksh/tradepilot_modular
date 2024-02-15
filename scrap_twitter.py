import pandas as pd
from ntscraper import Nitter
from datetime import datetime, timedelta, date
import os
import sqlite3

def create_db_table_if_not_exists(conn, stock_name):
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {stock_name} (
            Link TEXT,
            Text TEXT,
            Date TEXT,
            No_of_likes INTEGER,
            No_of_comments INTEGER
        )
    ''')
    conn.commit()


def insert_data_into_db(conn, data_list, stock_name):
    cursor = conn.cursor()
    for df in data_list:
        df.to_sql(stock_name, conn, if_exists='append', index=False)
    conn.commit()

def get_tweets(stock_name, mode, start_date, end_date, scraper, conn, date_counter):
    data_list = []  # List to store data for each day
    all_dates_skipped = True  # Set to True initially

    # Iterate over each day in the date range
    current_date = start_date

    while current_date <= end_date:
        current_date_str = current_date.strftime("%Y-%m-%d")
        next_date = current_date + pd.Timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%d")

        # Check if data for the current date already exists in the database
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {stock_name} WHERE Date = ?", (current_date_str,))
        count = cursor.fetchone()[0]
        print(f"The count of {current_date_str} is {count}")

        if count > 0:
            # Data for the current date already exists, skip this date
            print(f"Data for the date {current_date_str} already exists in the database. Skipping...\n")

        else:
            print(f"Data for the date {current_date_str} does not exist")

            try:
                # Retrieve tweets for the current day
                tweets, page_over = scraper.get_tweets(stock_name, mode=mode, since=current_date_str, until=next_date_str)
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

                if page_over and final_tweets:
                    print("tweets collected")
                    all_dates_skipped = False  # Set to False if data is not available in the db even for a single date

                    print(f"The page is over for date {current_date_str} and {len(final_tweets)} tweets are collected\n")

                    current_day_data = pd.DataFrame(final_tweets, columns=['Link', 'Text', 'Date', 'No_of_likes', 'No_of_comments'])
                    current_day_data = current_day_data.rename_axis('Index')
                    data_list.append(current_day_data)

                elif page_over:
                    print(f"The page is over for date {current_date_str} and No tweets are collected\n")

                else:
                    print(f"Some error happend during scraping of the date {current_date_str}\n")
                   

            except Exception as e:
                print(f"Error collecting tweets for the date {current_date_str}: {str(e)}")
                # Recreate the Nitter instance in case of an error
                scraper = Nitter(log_level=1, skip_instance_check=False)
                current_date = start_date
                continue

        # Move to the next day
        current_date += pd.Timedelta(days=1)

    # Insert data into SQLite database
    insert_data_into_db(conn, data_list, stock_name)

    return all_dates_skipped

stock_name = "RELIANCE"
db_filename = 'tweets_data.db'

start_date_str = "2024-01-17"
end_date_str = "2024-01-18"

start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
# end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')# Instantiate Nitter and SQLite connection outside the function

scraper = Nitter(log_level=1, skip_instance_check=False)
conn = sqlite3.connect(db_filename)


date_dict = {}

current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    date_dict[date_str] = 0
    current_date += timedelta(days=1)

# Create the table if it doesn't exist
create_db_table_if_not_exists(conn, stock_name)

# Keep calling the function until all dates are skipped
all_dates_skipped = False
while not all_dates_skipped:
    all_dates_skipped = get_tweets(stock_name, 'hashtag', start_date, end_date, scraper, conn, date_dict)

# Close the SQLite connection after all dates are collected
conn.close()

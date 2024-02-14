import pandas as pd
import sqlite3

db_filename = 'tweets_database.db'
conn = sqlite3.connect(db_filename)

current_date_str = "2023-12-30"
stock_name = "RELIANCE"

cursor = conn.cursor()
cursor.execute(f"SELECT COUNT(*) FROM {stock_name} WHERE Date = ?", (current_date_str,))
count = cursor.fetchone()[0]
print(f"The count of {current_date_str} is {count}")

conn.close()

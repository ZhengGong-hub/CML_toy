import os
import requests
import pandas as pd
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Example: Fetch some data from a public API
    response = requests.get('https://jsonplaceholder.typicode.com/posts')
    data = response.json()
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Display first few rows
    print("\nFirst 5 posts:")
    print(df.head())
    
    # Basic statistics
    print("\nBasic statistics:")
    print(f"Total number of posts: {len(df)}")
    print(f"Unique user IDs: {df['userId'].nunique()}")

if __name__ == "__main__":
    main() 
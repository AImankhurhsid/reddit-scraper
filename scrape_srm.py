import os
from dotenv import load_dotenv
import praw
import pandas as pd

def main():

    load_dotenv()


    client_id = os.getenv('client_id')
    client_secret = os.getenv('client_secret')
    username = os.getenv('username')
    password = os.getenv('password')
    user_agent = f"mac:reddit_karma_predictor:v1.0 (by u/{username})"

    if not all([client_id, client_secret, username, password]):
        print("Error: Missing one or more Reddit API credentials in .env")
        return
        
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=user_agent,
            check_for_async=False  
        )

        user = reddit.user.me()
        if user is None:
            print("Authentication failed: no user returned.")
            return
        print(f"Authenticated as: {user}")


        subreddit_name = 'SRMUNIVERSITY'  
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Fetching latest posts from r/{subreddit_name}...")

        limit = 100  
        posts = []

        for i, post in enumerate(subreddit.new(limit=limit), start=1):
            posts.append({
                'title': post.title,
                'score': post.score,
                'created_utc': post.created_utc,
                'num_comments': post.num_comments,
                'author': str(post.author),
                'permalink': f"https://reddit.com{post.permalink}"
            })
            print(f"{i}: {post.title[:60]}...")

        filename = f"{subreddit_name}_posts.csv"
        pd.DataFrame(posts).to_csv(filename, index=False)
        print(f"Saved {len(posts)} posts to {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

import os
import argparse
from dotenv import load_dotenv
import praw
import pandas as pd
import json
from deep_translator import GoogleTranslator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Scrape Reddit posts from specified subreddits')
    parser.add_argument('--subreddit', type=str, help='Subreddit name to scrape')
    parser.add_argument('--config', type=str, help='Path to config file', default='config.json')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'hi', 'es', 'fr', 'ta'], help='Target language for translation (en, hi, es, fr, ta)')
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"subreddits": ["SRMUNIVERSITY"], "limit": 100}

def translate_text(text, target_lang='en'):
    if target_lang == 'en' or not text:
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"Could not translate text: '{text[:30]}...'. Error: {e}")
        return text # Return original text on failure

def validate_post(post_data):
    required_fields = ['title', 'score', 'author']
    return all(post_data.get(field) is not None for field in required_fields)

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


        
        args = parse_arguments()
        config = load_config(args.config)
        subreddit_name = args.subreddit or config['subreddits'][0]
        limit = config.get('limit', 100)
        target_lang = args.lang

        subreddit = reddit.subreddit(subreddit_name)
        print(f"Fetching latest posts from r/{subreddit_name}...")

        posts = []
        for i, post in enumerate(subreddit.new(limit=limit), start=1):
            
            original_title = post.title
            translated_title = translate_text(original_title, target_lang)
            
            # Get post content (selftext for text posts, empty for link posts)
            original_content = post.selftext if post.is_self else ""
            translated_content = translate_text(original_content, target_lang) if original_content else ""

            post_data = {
                'title': original_title,
                f'title_{target_lang}': translated_title,
                'content': original_content,
                f'content_{target_lang}': translated_content,
                'score': post.score,
                'created_utc': post.created_utc,
                'num_comments': post.num_comments,
                'author': str(post.author),
                'permalink': f"https://reddit.com{post.permalink}",
                'upvote_ratio': post.upvote_ratio,
                'flair': post.link_flair_text,
                'is_original_content': post.is_original_content,
                'is_self': post.is_self,
                'over_18': post.over_18,
                'spoiler': post.spoiler,
                'stickied': post.stickied
            }
           
            if validate_post(post_data):
                posts.append(post_data)
                print(f"{i}: {post.title[:60]}...")
            else:
                print(f"Skipping post {i} due to missing required data")

        filename = f"{subreddit_name}_posts.csv"
        df = pd.DataFrame(posts)
        
        df = df.dropna(subset=['title', 'score'])  
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} valid posts to {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

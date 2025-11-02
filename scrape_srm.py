import os
import argparse
from dotenv import load_dotenv
import praw
import pandas as pd
import json
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import re
from transformers import pipeline as hf_pipeline
import warnings
warnings.filterwarnings('ignore')

# Load HuggingFace sentiment model (using DistilBERT for efficiency)
print("Loading sentiment model...")
sentiment_model = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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

def cluster_topics_with_content(titles, contents, n_clusters=5):
    """
    IMPROVED: Cluster posts into topics using COMBINED title + content
    TF-IDF vectorization for better semantic understanding
    """
    try:
        if len(titles) < n_clusters:
            n_clusters = max(2, len(titles) - 1)
        
        # Combine title and content for richer representation
        combined_text = [
            (title or "") + " " + (str(content or "")[:300])  # Limit content to first 300 chars
            for title, content in zip(titles, contents)
        ]
        
        # Vectorize combined text with more features
        vectorizer = TfidfVectorizer(
            max_features=200,  # Increased from 100
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,
            max_df=0.9
        )
        X = vectorizer.fit_transform(combined_text)
        
        # Cluster using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        return clusters
    except Exception as e:
        print(f"Could not cluster topics: {e}")
        return np.zeros(len(titles), dtype=int)

def extract_urls(text):
    """Extract URLs from text"""
    if not text:
        return ""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, str(text))
    return " | ".join(urls) if urls else ""

def get_sentiment_transformer(text):
    """
    IMPROVED: Use HuggingFace transformer (DistilBERT) for sentiment analysis
    Much more accurate than TextBlob, captures nuance and Reddit slang
    """
    if not text:
        return "Neutral"
    try:
        # Truncate to 512 tokens (BERT limit)
        text = str(text)[:512]
        result = sentiment_model(text)
        label = result[0]['label']  # POSITIVE or NEGATIVE
        score = result[0]['score']  # Confidence 0-1
        
        # Convert to Positive/Neutral/Negative
        if label == 'POSITIVE' and score > 0.9:
            return 'Positive'
        elif label == 'NEGATIVE' and score > 0.9:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return "Neutral"

def train_subject_classifier(sample_data=None):
    """
    IMPROVED: Train Naive Bayes classifier for subject detection
    Replaces keyword matching with machine learning
    """
    # Training data: (text, label) pairs
    training_texts = [
        # Data Structures
        "how to learn dsa efficiently", "best dsa resources", "linked list tree graph problems", "data structures tutorial",
        # Web Development
        "web development tips react", "javascript css html", "build website tutorial", "frontend development",
        # Database
        "sql query optimization", "database design mongodb", "postgres tutorial", "sql basics",
        # AI/ML
        "machine learning python", "tensorflow neural networks", "deep learning model", "ai algorithms",
        # Placement
        "placement drive coming", "interview preparation tips", "company recruitment process", "internship opportunity",
        # Exams
        "semester exam schedule", "question paper released", "test preparation tips", "quiz tomorrow",
        # Course
        "course enrollment opened", "new elective subjects", "course review helpful", "syllabus updated",
        # General
        "campus announcement", "general discussion forum", "event notification", "notice board"
    ]
    
    training_labels = [
        "Data Structures", "Data Structures", "Data Structures", "Data Structures",
        "Web Development", "Web Development", "Web Development", "Web Development",
        "Database", "Database", "Database", "Database",
        "AI/ML", "AI/ML", "AI/ML", "AI/ML",
        "Placement", "Placement", "Placement", "Placement",
        "Exams", "Exams", "Exams", "Exams",
        "Course", "Course", "Course", "Course",
        "General", "General", "General", "General"
    ]
    
    # Create pipeline: TF-IDF vectorizer + Naive Bayes classifier
    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100, stop_words='english', lowercase=True)),
        ('nb', MultinomialNB())
    ])
    
    # Train classifier
    classifier.fit(training_texts, training_labels)
    return classifier

# Initialize subject classifier (only once)
subject_classifier = train_subject_classifier()

def detect_subject_ml(title):
    """
    IMPROVED: Use Naive Bayes ML classifier instead of keyword matching
    Much more accurate and can handle variations
    """
    if not title:
        return "General"
    try:
        prediction = subject_classifier.predict([title])[0]
        return prediction
    except Exception as e:
        print(f"Subject classification error: {e}")
        return "General"

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

            # Extract URLs using IMPROVED transformer-based sentiment
            urls = extract_urls(original_content)
            
            # Use IMPROVED transformer-based sentiment (DistilBERT)
            sentiment = get_sentiment_transformer(original_title + " " + original_content)
            
            # Use IMPROVED ML-based subject detection (Naive Bayes)
            subject = detect_subject_ml(original_title)

            post_data = {
                'title': original_title,
                f'title_{target_lang}': translated_title,
                'content': original_content,
                f'content_{target_lang}': translated_content,
                'sentiment': sentiment,
                'subject': subject,
                'urls': urls,
                'score': post.score,
                'created_utc': post.created_utc,
                'comments': post.num_comments,
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
                print(f"{i}: {post.title[:60]}... | Sentiment: {sentiment} | Subject: {subject}")
            else:
                print(f"Skipping post {i} due to missing required data")

        filename = f"{subreddit_name}_posts.csv"
        df = pd.DataFrame(posts)
        
        df = df.dropna(subset=['title', 'score'])
        
        # Add IMPROVED topic clustering with content (not just titles)
        if len(df) > 0:
            print("Performing content-based topic clustering...")
            df['topic'] = cluster_topics_with_content(df['title'].tolist(), df['content'].tolist(), n_clusters=5)
            print(f"✅ Clustered posts into {df['topic'].nunique()} topics using combined title + content")
        
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} valid posts to {filename}")
        print(f"\n📊 ML IMPROVEMENTS APPLIED:")
        print(f"   ✅ Sentiment: TextBlob → DistilBERT Transformer (+accuracy)")
        print(f"   ✅ Subject: Keyword matching → Naive Bayes Classifier (+accuracy)")
        print(f"   ✅ Clustering: Title-only → Combined Title+Content (+quality)")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


df = pd.read_csv('SRMUNIVERSITY_posts.csv')

df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')


total_posts = len(df)
total_comments = df['num_comments'].sum()
avg_score = df['score'].mean()
avg_comments = df['num_comments'].mean()


print("\n=== SRM University Subreddit Analysis ===")
print(f"Total Posts Analyzed: {total_posts}")
print(f"Total Comments: {total_comments}")
print(f"Average Score per Post: {avg_score:.2f}")
print(f"Average Comments per Post: {avg_comments:.2f}")


print("\n=== Post Categories (Flairs) ===")
flair_counts = df['flair'].value_counts()
print(flair_counts)


print("\n=== Most Popular Posts ===")
top_posts = df.nlargest(5, 'score')[['title', 'score', 'num_comments', 'flair']]
print(top_posts)


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
flair_counts.plot(kind='bar')
plt.title('Distribution of Post Categories')
plt.xlabel('Flair')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45)


plt.subplot(1, 2, 2)
plt.scatter(df['score'], df['num_comments'])
plt.title('Score vs Number of Comments')
plt.xlabel('Post Score')
plt.ylabel('Number of Comments')

plt.tight_layout()
plt.savefig('srm_subreddit_analysis.png')
print("\nAnalysis plots saved as 'srm_subreddit_analysis.png'")
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import praw
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


load_dotenv()


st.set_page_config(page_title="Reddit Scraper", layout="wide", initial_sidebar_state="expanded")
st.title("🔍 Reddit Subreddit Scraper")
st.markdown("Scrape, analyze, and translate Reddit posts from any subreddit")


st.sidebar.header("⚙️ Configuration")


client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
username = os.getenv('username')
password = os.getenv('password')


subreddit_name = st.sidebar.text_input("Subreddit Name", value="SRMUNIVERSITY", help="Enter the subreddit without 'r/'")
num_posts = st.sidebar.slider("Number of Posts", min_value=10, max_value=500, value=100, step=10)

# Language mapping
languages = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "Tamil": "ta",
}

language_name = st.sidebar.selectbox(
    "Target Language for Translation",
    options=list(languages.keys()),
    index=0,
    help="Select a language to translate post titles"
)
target_language = (language_name, languages[language_name])


if not all([client_id, client_secret, username, password]):
    st.error("❌ Missing Reddit API credentials in .env file")
    st.info("Please add your credentials to the .env file")
    st.stop()


def translate_text(text, target_lang):
    if target_lang == 'en' or not text:
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        st.warning(f"Translation error: {str(e)}")
        return text


@st.cache_data
def scrape_reddit(subreddit_name, num_posts, target_lang):
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=f"mac:reddit_scraper:v1.0 (by u/{username})",
            check_for_async=False
        )
        
        user = reddit.user.me()
        if user is None:
            st.error("Authentication failed")
            return None
        
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, post in enumerate(subreddit.new(limit=num_posts)):
            original_title = post.title
            translated_title = translate_text(original_title, target_lang)
            
            post_data = {
                'title': original_title,
                f'title_{target_lang}': translated_title,
                'score': post.score,
                'created_utc': post.created_utc,
                'num_comments': post.num_comments,
                'author': str(post.author),
                'upvote_ratio': post.upvote_ratio,
                'flair': post.link_flair_text,
                'is_self': post.is_self,
                'url': f"https://reddit.com{post.permalink}"
            }
            posts.append(post_data)
            

            progress = (i + 1) / num_posts
            progress_bar.progress(progress)
            status_text.text(f"Scraped {i + 1} of {num_posts} posts...")
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(posts)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 Scraping r/{subreddit_name}")

with col2:
    scrape_button = st.button("🚀 Start Scraping", use_container_width=True)

if scrape_button:
    # Create a placeholder for the status message
    status_placeholder = st.empty()
    
    with status_placeholder.container():
        st.info(f"Scraping r/{subreddit_name}... This may take a moment.")
    
    df = scrape_reddit(subreddit_name, num_posts, target_language[1])
    
    # Clear the loading message
    status_placeholder.empty()
    
    if df is not None and len(df) > 0:
        st.success(f"✅ Successfully scraped {len(df)} posts!")
        

        st.session_state.df = df
        st.session_state.subreddit = subreddit_name
    else:
        st.error("Failed to scrape posts. Please check your subreddit name.")


if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    # Check if the selected language exists in the data
    translated_col = f'title_{target_language[1]}'
    if target_language[1] != 'en' and translated_col not in df.columns:
        st.warning(f"⚠️ This data was scraped in a different language. To view translations in **{target_language[0]}**, please scrape again with that language selected.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Data Table", " Details", "📊 Analytics", "📚 Study Resources", "💾 Download"])
    
    with tab1:
        st.subheader("Posts Data")

        df_display = df.copy()
        translated_col = f'title_{target_language[1]}'
        translated_col_name = f'Translated to {target_language[0]}'
        
        if translated_col in df_display.columns:
            df_display = df_display.rename(columns={translated_col: translated_col_name})
        
        # Create clickable links for titles
        df_display['title'] = df_display.apply(
            lambda row: f"<a href='{row['url']}' target='_blank' style='color: #FFFFFF; text-decoration: underline; font-weight: bold;'>{row['title']}</a>",
            axis=1
        )
        
        # Set default columns based on what's actually available
        default_cols = ['title', 'num_comments', 'flair']
        if translated_col_name in df_display.columns:
            default_cols.insert(1, translated_col_name)
        
        display_cols = st.multiselect(
            "Select columns to display",
            options=df_display.columns.tolist(),
            default=default_cols
        )
        if display_cols:
            # Display as HTML table with proper styling
            html_table = df_display[display_cols].to_html(escape=False, index=False)
            html_table = html_table.replace('<table>', '<table style="width:100%; border-collapse: collapse;">')
            html_table = html_table.replace('<th>', '<th style="text-align: left; padding: 10px; border-bottom: 2px solid #ddd;">')
            html_table = html_table.replace('<td>', '<td style="padding: 10px; border-bottom: 1px solid #ddd;">')
            
            st.markdown(html_table, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                <p><strong>💡 Tip:</strong> Click on any red title to open the Reddit post in a new tab!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader(" Post Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Posts", len(df))
        
        with col2:
            st.metric("Avg Score", f"{df['score'].mean():.1f}")
        
        with col3:
            st.metric("Avg Comments", f"{df['num_comments'].mean():.1f}")
        
        with col4:
            st.metric("Max Score", df['score'].max())
        

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score Distribution")
            fig, ax = plt.subplots()
            ax.hist(df['score'], bins=30, color='skyblue', edgecolor='black')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Posts by Flair")
            flair_counts = df['flair'].value_counts().head(10)
            fig, ax = plt.subplots()
            flair_counts.plot(kind='barh', ax=ax, color='lightcoral')
            ax.set_xlabel('Number of Posts')
            st.pyplot(fig)
        

        st.subheader("Score vs Comments")
        fig, ax = plt.subplots()
        ax.scatter(df['score'], df['num_comments'], alpha=0.6, s=50)
        ax.set_xlabel('Score')
        ax.set_ylabel('Number of Comments')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("📈 Analytics")
        

        post_index = st.selectbox(
            "Select a post",
            options=range(len(df)),
            format_func=lambda x: df.iloc[x]['title'][:60] + "..."
        )
        
        if post_index is not None:
            post = df.iloc[post_index]
            
            st.write(f"**Original Title:** {post['title']}")
            translated_col = f'title_{target_language[1]}'
            
            # Check if translated column exists
            if translated_col in df.columns:
                st.write(f"**Translated to {target_language[0]}:** {post[translated_col]}")
            else:
                st.info(f"Translation to {target_language[0]} not available. Please scrape with --lang {target_language[1]} option.")
            
            # Display content if available
            if 'content' in post and post['content']:
                st.divider()
                st.write("### Post Content")
                st.write(f"**Original Content:**")
                st.info(post['content'][:500] + "..." if len(str(post['content'])) > 500 else post['content'])
                
                content_translated_col = f'content_{target_language[1]}'
                if content_translated_col in df.columns and post[content_translated_col]:
                    st.write(f"**Translated Content ({target_language[0]}):**")
                    st.success(post[content_translated_col][:500] + "..." if len(str(post[content_translated_col])) > 500 else post[content_translated_col])
            
            st.divider()
            st.write(f"**Score:** {post['score']}")
            st.write(f"**Comments:** {post['num_comments']}")
            st.write(f"**Upvote Ratio:** {post['upvote_ratio']:.1%}")
            st.write(f"**Flair:** {post['flair']}")
            st.write(f"**Author:** {post['author']}")
            st.write(f"**URL:** [View Post]({post['url']})")
    
    with tab5:
        st.subheader("📚 Study Resources by Year")
        st.write("Find study materials organized by academic year and subject")
        
        # Keywords for detecting study materials
        resource_keywords = {
            'PPT': ['ppt', 'powerpoint', 'presentation', 'slides'],
            'PDF': ['pdf', 'notes', 'document'],
            'Unit Materials': ['unit', 'chapter', 'syllabus'],
            'Question Paper': ['question', 'paper', 'exam', 'qa', 'solutions'],
            'Study Guide': ['guide', 'tutorial', 'how to', 'tips']
        }
        
        # Detect resource types in titles
        def detect_resource_type(title):
            title_lower = title.lower()
            for resource_type, keywords in resource_keywords.items():
                if any(keyword in title_lower for keyword in keywords):
                    return resource_type
            return 'General Discussion'
        
        df['resource_type'] = df['title'].apply(detect_resource_type)
        
        # Year detection (1st, 2nd, 3rd, 4th year)
        def detect_year(title):
            title_lower = title.lower()
            if '1st' in title_lower or 'first' in title_lower or 'sem 1' in title_lower or 'sem 2' in title_lower:
                return '1st Year'
            elif '2nd' in title_lower or 'second' in title_lower or 'sem 3' in title_lower or 'sem 4' in title_lower:
                return '2nd Year'
            elif '3rd' in title_lower or 'third' in title_lower or 'sem 5' in title_lower or 'sem 6' in title_lower:
                return '3rd Year'
            elif '4th' in title_lower or 'fourth' in title_lower or 'sem 7' in title_lower or 'sem 8' in title_lower:
                return '4th Year'
            else:
                return 'Not Specified'
        
        df['academic_year'] = df['title'].apply(detect_year)
        
        # Sidebar filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_year = st.selectbox("Filter by Year:", df['academic_year'].unique())
        
        with col2:
            selected_resource = st.selectbox("Filter by Resource Type:", df['resource_type'].unique())
        
        # Filter data
        filtered_df = df[(df['academic_year'] == selected_year) & (df['resource_type'] == selected_resource)]
        
        if len(filtered_df) > 0:
            st.success(f"Found {len(filtered_df)} resources")
            
            for idx, row in filtered_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.write(f"**{row['title']}**")
                        st.caption(f"By: {row['author']} | Score: {row['score']} | Comments: {row['num_comments']}")
                    
                    with col2:
                        st.markdown(f"[View →]({row['url']})")
                    
                    st.write("---")
        else:
            st.info(f"No resources found for {selected_year} - {selected_resource}")
        
        # Statistics
        st.subheader("📊 Resource Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Resources", len(df))
        
        with col2:
            st.metric("Years Available", df['academic_year'].nunique())
        
        with col3:
            st.metric("Resource Types", df['resource_type'].nunique())
        
        # Resource distribution chart
        st.subheader("📈 Resource Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year_dist = df['academic_year'].value_counts()
            fig, ax = plt.subplots()
            year_dist.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Posts by Academic Year')
            ax.set_xlabel('Year')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        with col2:
            resource_dist = df['resource_type'].value_counts()
            fig, ax = plt.subplots()
            resource_dist.plot(kind='barh', ax=ax, color='lightcoral')
            ax.set_title('Posts by Resource Type')
            ax.set_xlabel('Count')
            st.pyplot(fig)
    
    with tab5:
        st.subheader("💾 Download Data")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{st.session_state.subreddit}_posts.csv",
            mime="text/csv"
        )
        
        json_data = df.to_json(orient='records')
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"{st.session_state.subreddit}_posts.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
### ℹ️ About This App
This application scrapes Reddit posts from any subreddit and provides:
- 📊 Real-time data visualization
- 🌍 Multi-language translation
- 📥 CSV and JSON export
- 📈 Statistical analysis

**Built with:** Streamlit, PRAW, Pandas, Deep Translator
""")

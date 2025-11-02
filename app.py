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


subreddit_name = st.sidebar.text_input("Subreddit Name", value="SRMUNIVERSITY")
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
    index=0
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Data Table", " Details", "📊 Analytics", "📚 Study Resources", "💡 Insights", "💾 Download"])
    
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
        if 'topic' in df_display.columns:
            default_cols.insert(1, 'topic')
        if translated_col_name in df_display.columns:
            default_cols.insert(2, translated_col_name)
        
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
    
    with tab2:
        st.subheader(" Post Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Posts", len(df))
        
        with col2:
            st.metric("Avg Score", f"{df['score'].mean():.1f}")
        
        with col3:
            st.metric("Total Comments", int(df['num_comments'].sum()))
        
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
        
        # Topic distribution if available
        if 'topic' in df.columns:
            st.subheader("Posts by Topic")
            topic_counts = df['topic'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar([f'Topic {i}' for i in topic_counts.index], topic_counts.values, color='mediumpurple')
            ax.set_ylabel('Number of Posts')
            st.pyplot(fig)
            st.divider()
        

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
        st.write("### Filter Resources")
        filter_option = st.radio("Filter by:", ["Resource Type", "Academic Year", "View All"], horizontal=True)
        
        if filter_option == "Resource Type":
            selected_filter = st.selectbox("Select Resource Type:", sorted(df['resource_type'].unique()))
            filtered_df = df[df['resource_type'] == selected_filter]
            filter_label = selected_filter
        elif filter_option == "Academic Year":
            selected_filter = st.selectbox("Select Year:", sorted(df['academic_year'].unique()))
            filtered_df = df[df['academic_year'] == selected_filter]
            filter_label = selected_filter
        else:
            filtered_df = df.sort_values('score', ascending=False)
            filter_label = "All Resources"
        
        if len(filtered_df) > 0:
            st.success(f"Found {len(filtered_df)} resources in {filter_label}")
            
            for idx, row in filtered_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.write(f"**{row['title']}**")
                        year_badge = f"📅 {row['academic_year']}" if row['academic_year'] != 'Not Specified' else ""
                        type_badge = f"📚 {row['resource_type']}"
                        st.caption(f"By: {row['author']} | Score: {row['score']} | Comments: {row['num_comments']} | {type_badge} {year_badge}")
                    
                    with col2:
                        post_url = row['permalink'] if 'permalink' in row else row['url'] if 'url' in row else '#'
                        st.markdown(f"[View →]({post_url})")
                    
                    st.write("---")
        else:
            st.info(f"No resources found in {filter_label}")
        
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
        st.subheader("💡 Insights & Analytics")
        
        # Add semester detection
        def detect_semester(title):
            """Detect semester from title"""
            title_lower = title.lower()
            # Check for explicit semester mentions
            if 'sem 1' in title_lower or 'semester 1' in title_lower:
                return 'Sem 1'
            elif 'sem 2' in title_lower or 'semester 2' in title_lower:
                return 'Sem 2'
            elif 'sem 3' in title_lower or 'semester 3' in title_lower:
                return 'Sem 3'
            elif 'sem 4' in title_lower or 'semester 4' in title_lower:
                return 'Sem 4'
            elif 'sem 5' in title_lower or 'semester 5' in title_lower:
                return 'Sem 5'
            elif 'sem 6' in title_lower or 'semester 6' in title_lower:
                return 'Sem 6'
            elif 'sem 7' in title_lower or 'semester 7' in title_lower:
                return 'Sem 7'
            elif 'sem 8' in title_lower or 'semester 8' in title_lower:
                return 'Sem 8'
            # Check for 1st/2nd/3rd/4th year mentions
            elif '1st' in title_lower or 'first year' in title_lower:
                return 'Sem 1-2'
            elif '2nd' in title_lower or 'second year' in title_lower:
                return 'Sem 3-4'
            elif '3rd' in title_lower or 'third year' in title_lower:
                return 'Sem 5-6'
            elif '4th' in title_lower or 'fourth year' in title_lower:
                return 'Sem 7-8'
            else:
                return 'All Semesters'
        
        # Add subject detection with SRM curriculum
        def detect_subject_improved(title):
            """Detect subject from title based on SRM curriculum"""
            title_lower = title.lower()
            
            # 1st Year (Sem 1-2) - Engineering Fundamentals
            if any(x in title_lower for x in ['physics', 'chemistry', 'maths', 'calculus', 'differential equations', 'linear algebra']):
                return 'Math & Science'
            if any(x in title_lower for x in ['c programming', 'python', 'java basics', 'programming fundamentals']):
                return 'Programming Basics'
            
            # 2nd Year (Sem 3-4) - Core Engineering
            if any(x in title_lower for x in ['dld', 'digital', 'logic', 'circuit', 'electronics']):
                return 'Digital Logic & Circuits'
            if any(x in title_lower for x in ['dsa', 'data structures', 'arrays', 'linked list', 'tree', 'graph']):
                return 'Data Structures'
            if any(x in title_lower for x in ['oop', 'object oriented', 'java oop', 'oops']):
                return 'OOP & Java'
            if any(x in title_lower for x in ['database', 'sql', 'db', 'dbms', 'normalization']):
                return 'Database Systems'
            
            # 3rd Year (Sem 5-6) - Specialization
            if any(x in title_lower for x in ['web', 'html', 'css', 'javascript', 'react', 'angular', 'nodejs']):
                return 'Web Development'
            if any(x in title_lower for x in ['ai', 'machine learning', 'ml', 'deep learning', 'neural', 'nlp']):
                return 'AI & Machine Learning'
            if any(x in title_lower for x in ['os', 'operating system', 'linux', 'windows', 'process', 'memory']):
                return 'Operating Systems'
            if any(x in title_lower for x in ['cn', 'computer network', 'tcp', 'ip', 'networking', 'protocols']):
                return 'Computer Networks'
            
            # 4th Year (Sem 7-8) - Advanced & Placement
            if any(x in title_lower for x in ['placement', 'interview', 'placement drive', 'placement news', 'job', 'internship']):
                return 'Placement & Career'
            if any(x in title_lower for x in ['project', 'capstone', 'final year', 'fyp']):
                return 'Final Year Project'
            
            # General topics
            if any(x in title_lower for x in ['exam', 'test', 'qp', 'question paper', 'mock']):
                return 'Exams & Tests'
            if any(x in title_lower for x in ['cgpa', 'gpa', 'marks', 'grade', 'result', 'transcript']):
                return 'Academics & Results'
            if any(x in title_lower for x in ['branch', 'cse', 'ece', 'civil', 'mechanical', 'ae']):
                return 'Branch Related'
            
            return 'General Discussion'
        
        # Add columns for semester and improved subject
        df['semester'] = df['title'].apply(detect_semester)
        df['subject_category'] = df['title'].apply(detect_subject_improved)
        
        # Filter options
        st.write("### 🎯 Filter Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_semester = st.selectbox(
                "📚 Select Semester/Year:",
                sorted(df['semester'].unique()),
                index=0
            )
        
        with col2:
            selected_subject = st.selectbox(
                "📖 Select Subject Category:",
                sorted(df['subject_category'].unique()),
                index=0
            )
        
        # Filter data based on selection
        filtered_insights = df[(df['semester'] == selected_semester) & (df['subject_category'] == selected_subject)]
        
        st.info(f"Showing {len(filtered_insights)} posts for **{selected_semester}** - **{selected_subject}**")
        st.divider()
        
        # Extract URLs button if not already present
        if 'urls' not in df.columns:
            st.warning("⚠️ URL extraction not available in current data")
            if st.button("🔄 Extract URLs from Posts Now", key="extract_urls_btn"):
                with st.spinner("Extracting URLs..."):
                    import re
                    df['urls'] = df['content'].fillna('').apply(
                        lambda x: " | ".join(re.findall(r'https?://[^\s]+', str(x))) if x else ""
                    )
                    st.session_state.df = df
                    st.success("✅ URLs extracted!")
                    st.rerun()
        
        # Sentiment Analysis - for filtered data
        st.markdown("### 😊 Sentiment Analysis")
        if 'sentiment' in filtered_insights.columns:
            sentiment_counts = filtered_insights['sentiment'].value_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("😊 Positive Posts", sentiment_counts.get('Positive', 0))
            with col2:
                st.metric("😐 Neutral Posts", sentiment_counts.get('Neutral', 0))
            with col3:
                st.metric("😢 Negative Posts", sentiment_counts.get('Negative', 0))
            
            if len(sentiment_counts) > 0:
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'gray', 'red'])
                ax.set_ylabel('Number of Posts')
                ax.set_title('Sentiment Distribution')
                plt.xticks(rotation=0)
                st.pyplot(fig)
        
        # Subject Analysis
        st.markdown("### 📚 Posts by Subject")
        if 'subject' in filtered_insights.columns:
            subject_counts = filtered_insights['subject'].value_counts()
            if len(subject_counts) > 0:
                fig, ax = plt.subplots()
                subject_counts.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_xlabel('Number of Posts')
                ax.set_title('Subject Distribution')
                st.pyplot(fig)
                
                # Top subject details
                st.write("**Top Subjects in Selection:**")
                for subject, count in subject_counts.head(5).items():
                    st.write(f"• {subject}: {count} posts")
        
        # Top Contributors in filtered data
        st.markdown("### 👥 Top Contributors")
        if len(filtered_insights) > 0:
            top_authors = filtered_insights['author'].value_counts().head(10)
            if len(top_authors) > 0:
                fig, ax = plt.subplots()
                top_authors.plot(kind='barh', ax=ax, color='coral')
                ax.set_xlabel('Number of Posts')
                ax.set_title('Top Contributors in Selection')
                st.pyplot(fig)
        
        # Resources with Links - for filtered data
        st.markdown("### 🔗 Resources with Links")
        if 'urls' in filtered_insights.columns:
            resources_with_links = filtered_insights[filtered_insights['urls'].notna() & (filtered_insights['urls'] != '')]
            if len(resources_with_links) > 0:
                st.write(f"Found **{len(resources_with_links)} posts** with resource links")
                for idx, row in resources_with_links.head(10).iterrows():
                    st.write(f"**{row.get('title','')[:60]}...**")
                    # Safe access for optional fields
                    sentiment = row.get('sentiment', 'N/A') if hasattr(row, 'get') else (row['sentiment'] if 'sentiment' in row else 'N/A')
                    author = row.get('author', 'unknown') if hasattr(row, 'get') else row.get('author', 'unknown')
                    score = row.get('score', '') if hasattr(row, 'get') else row.get('score', '')
                    st.caption(f"Author: {author} | Score: {score} | {sentiment}")
                    st.write(f"🔗 {row.get('urls', '')}")
                    st.divider()
            else:
                st.info("No resources with links found in this selection")
        else:
            st.info("No 'urls' column in data — scrape with the latest scraper to enable link extraction")
    
    with tab6:
        
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

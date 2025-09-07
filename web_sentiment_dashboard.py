import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Customer Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    # Load data
    df = pd.read_csv('/mnt/c/Users/prava/Downloads/amazon.csv')
    
    # Process sentiment
    df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
    df['review_length'] = df['review_content'].str.len()
    
    def analyze_sentiment(row):
        rating = row['rating_numeric']
        content = str(row['review_content']).lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'awesome', 'satisfied', 'recommend']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappointed', 'useless', 'defective', 'broken']
        
        pos_count = sum(1 for word in positive_words if word in content)
        neg_count = sum(1 for word in negative_words if word in content)
        
        if rating >= 4 and pos_count > neg_count:
            return 'Positive'
        elif rating <= 2 or neg_count > pos_count:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment'] = df.apply(analyze_sentiment, axis=1)
    
    def sentiment_score(row):
        if row['sentiment'] == 'Positive':
            return row['rating_numeric'] / 5.0
        elif row['sentiment'] == 'Negative':
            return (5 - row['rating_numeric']) / 5.0 * -1
        else:
            return 0
    
    df['sentiment_score'] = df.apply(sentiment_score, axis=1)
    return df

# Load data
df = load_and_process_data()

# Header
st.markdown('<h1 class="main-header">🎯 Customer Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Reviews", f"{len(df):,}")

with col2:
    avg_rating = df['rating_numeric'].mean()
    st.metric("⭐ Average Rating", f"{avg_rating:.2f}/5.0")

with col3:
    positive_pct = (df['sentiment'] == 'Positive').mean() * 100
    st.metric("😊 Satisfaction Rate", f"{positive_pct:.1f}%")

with col4:
    avg_sentiment = df['sentiment_score'].mean()
    st.metric("📈 Sentiment Score", f"{avg_sentiment:.2f}/1.0")

st.divider()

# Charts Row 1
col1, col2 = st.columns(2)

with col1:
    # Sentiment Distribution Pie Chart
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'Positive': '#6BCF7F', 'Neutral': '#FFD93D', 'Negative': '#FF6B6B'}
    
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Overall Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Sentiment by Rating
    sentiment_rating = pd.crosstab(df['rating_numeric'], df['sentiment'])
    
    fig_bar = px.bar(
        sentiment_rating,
        title="Sentiment by Star Rating",
        color_discrete_map=colors,
        labels={'value': 'Count', 'index': 'Star Rating'}
    )
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

# Charts Row 2
col1, col2 = st.columns(2)

with col1:
    # Top Categories Sentiment
    top_categories = df['category'].str.split('|').str[0].value_counts().head(5).index
    cat_sentiment = df[df['category'].str.split('|').str[0].isin(top_categories)]
    cat_sentiment_pct = pd.crosstab(cat_sentiment['category'].str.split('|').str[0], 
                                   cat_sentiment['sentiment'], normalize='index') * 100
    
    fig_cat = px.bar(
        cat_sentiment_pct,
        orientation='h',
        title="Sentiment % by Top Categories",
        color_discrete_map=colors,
        labels={'value': 'Percentage', 'index': 'Category'}
    )
    fig_cat.update_layout(height=400)
    st.plotly_chart(fig_cat, use_container_width=True)

with col2:
    # Sentiment Score Distribution
    fig_hist = px.histogram(
        df,
        x='sentiment_score',
        title="Sentiment Score Distribution",
        nbins=25,
        color_discrete_sequence=['#4ECDC4']
    )
    fig_hist.add_vline(x=df['sentiment_score'].mean(), line_dash="dash", 
                       line_color="red", annotation_text=f"Mean: {df['sentiment_score'].mean():.2f}")
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# Charts Row 3
col1, col2 = st.columns(2)

with col1:
    # Review Length by Sentiment
    sentiment_length = df.groupby('sentiment')['review_length'].mean().reset_index()
    
    fig_length = px.bar(
        sentiment_length,
        x='sentiment',
        y='review_length',
        title="Average Review Length by Sentiment",
        color='sentiment',
        color_discrete_map=colors
    )
    fig_length.update_layout(height=400)
    st.plotly_chart(fig_length, use_container_width=True)

with col2:
    # Summary Statistics
    st.subheader("📋 Detailed Summary")
    
    total_reviews = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    
    # Most positive/negative categories
    cat_sentiment_avg = df.groupby(df['category'].str.split('|').str[0])['sentiment_score'].mean()
    most_positive_cat = cat_sentiment_avg.idxmax()
    most_negative_cat = cat_sentiment_avg.idxmin()
    
    st.write(f"""
    **Sentiment Breakdown:**
    - 🟢 Positive: {sentiment_counts.get('Positive', 0):,} ({sentiment_counts.get('Positive', 0)/total_reviews*100:.1f}%)
    - 🟡 Neutral: {sentiment_counts.get('Neutral', 0):,} ({sentiment_counts.get('Neutral', 0)/total_reviews*100:.1f}%)
    - 🔴 Negative: {sentiment_counts.get('Negative', 0):,} ({sentiment_counts.get('Negative', 0)/total_reviews*100:.1f}%)
    
    **Category Insights:**
    - 👍 Most Positive: {most_positive_cat}
    - 👎 Most Critical: {most_negative_cat}
    
    **Review Characteristics:**
    - Average review length: {df['review_length'].mean():.0f} characters
    - Products analyzed: {df['product_id'].nunique():,}
    - Categories covered: {df['category'].str.split('|').str[0].nunique()}
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    📊 Customer Sentiment Analysis Dashboard | Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
import base64
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Airline Sentiment Suite",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NLTK DATA ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
download_nltk_data()

# --- BACKGROUND IMAGE ---
@st.cache_data
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"File not found: {file}. Please make sure you saved the image as 'background.jpg' in the project folder.")
        return ""

img = get_img_as_base64("background.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] {{
background-color: rgba(255,255,255,0.5);
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# --- LOAD MODEL AND VECTORIZER ---
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('svm_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
# First, get the regular list of boring words
default_stopwords = set(stopwords.words('english'))

# Now, tell our program which words are important and should NOT be ignored
words_to_keep = {'not', 'no', 'nor', 'against'}

# Create the final, smart ignore list by taking the old list and removing our important words
stop_words = default_stopwords - words_to_keep

# --- HELPER FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\s]", '', text)
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in stop_words])

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    try:
        prediction_proba = model.predict_proba(vectorized_text)
        confidence = prediction_proba.max()
    except AttributeError:
        confidence = None
    return prediction, confidence

# --- PAGE 1: SENTIMENT PREDICTOR ---
def page_predictor():
    st.title("✈️ Airline Tweet Sentiment Predictor")
    st.markdown("Analyze a single tweet or upload a CSV file for batch analysis.")

    tab1, tab2 = st.tabs(["Single Tweet Analysis", "Batch Analysis (CSV)"])

    # Single Tweet Tab
    with tab1:
        st.subheader("Analyze a single tweet")
        example_tweet = st.selectbox(
            "Choose an example tweet, or type your own below:",
            ("Select an example...",
             "United, you are the best! Smooth flight and great service.",
             "American Airlines, my flight was delayed for 3 hours. Not happy.",
             "Just landed with Delta. The flight was okay, nothing special."),
            key='examples'
        )

        if example_tweet != "Select an example...":
            tweet_input = st.text_area("Your tweet:", value=example_tweet, height=150, key='text_area_single')
        else:
            tweet_input = st.text_area("Your tweet:", placeholder="Type your tweet here...", height=150, key='text_area_custom')

        if st.button("Analyze Sentiment", key='analyze_single'):
            if tweet_input.strip():
                with st.spinner('Analyzing...'):
                    prediction, confidence = predict_sentiment(tweet_input)

                sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😠"}
                sentiment_color = {"positive": "green", "neutral": "blue", "negative": "red"}

                st.markdown(f"### Analysis Result")
                st.markdown(f"**Sentiment:** <span style='color:{sentiment_color[prediction]}; font-size: 24px;'>{prediction.capitalize()} {sentiment_emoji[prediction]}</span>", unsafe_allow_html=True)
                if confidence:
                    st.metric(label="Confidence", value=f"{confidence:.2%}")
                    st.progress(confidence)
            else:
                st.warning("Please enter a tweet.")

    # Batch Analysis Tab
    with tab2:
        st.subheader("Upload a CSV for batch analysis")
        st.markdown("Your CSV should have a column named 'text' or 'tweets' containing the tweets to analyze.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                tweet_col = 'text' if 'text' in df.columns else 'tweets' if 'tweets' in df.columns else None

                if not tweet_col:
                    st.error("CSV must contain a 'text' or 'tweets' column.")
                else:
                    st.success(f"Found '{tweet_col}' column. Ready to process {len(df)} tweets.")
                    if st.button("Start Batch Analysis", key='analyze_batch'):
                        progress_bar = st.progress(0)
                        results = []
                        for i, row in df.iterrows():
                            prediction, _ = predict_sentiment(row[tweet_col])
                            results.append(prediction)
                            progress_bar.progress((i + 1) / len(df))

                        df['predicted_sentiment'] = results
                        st.balloons()
                        st.success("Batch analysis complete!")
                        st.dataframe(df)

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download results as CSV",
                            data=csv,
                            file_name='sentiment_predictions.csv',
                            mime='text/csv',
                        )
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- PAGE 2: DATA & MODEL INSIGHTS ---
def page_insights():
    st.title("📊 Data & Model Insights")
    st.markdown("Explore the dataset that was used to train the model. **You will need the original `Tweets.csv` file for this.**")

    uploaded_file = st.file_uploader("Upload the original Tweets.csv file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded `Tweets.csv` with {len(df)} rows.")

            # --- Visualizations ---
            st.header("Sentiment Distribution")
            fig_sentiment = px.bar(df['airline_sentiment'].value_counts().reset_index(),
                                   x='index', y='airline_sentiment',
                                   labels={'index':'Sentiment', 'airline_sentiment':'Number of Tweets'},
                                   color='index', color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'},
                                   template='plotly_white')
            st.plotly_chart(fig_sentiment, use_container_width=True)

            st.header("Tweet Distribution by Airline")
            fig_airline = px.pie(df, names='airline',
                                 title='Percentage of Tweets per Airline',
                                 hole=.3,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_airline, use_container_width=True)

            st.header("Sentiment Word Clouds")
            st.markdown("These word clouds show the most frequent words for each sentiment after cleaning.")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Positive")
                positive_text = " ".join([clean_text(text) for text in df[df.airline_sentiment == 'positive']['text']])
                wordcloud = WordCloud(stopwords=stop_words, background_color="rgba(255, 255, 255, 0)", colormap='Greens', max_words=100).generate(positive_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            with col2:
                st.subheader("Neutral")
                neutral_text = " ".join([clean_text(text) for text in df[df.airline_sentiment == 'neutral']['text']])
                wordcloud = WordCloud(stopwords=stop_words, background_color="rgba(255, 255, 255, 0)", colormap='Blues', max_words=100).generate(neutral_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            with col3:
                st.subheader("Negative")
                negative_text = " ".join([clean_text(text) for text in df[df.airline_sentiment == 'negative']['text']])
                wordcloud = WordCloud(stopwords=stop_words, background_color="rgba(255, 255, 255, 0)", colormap='Reds', max_words=100).generate(negative_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Failed to process the uploaded file. Please ensure it's the correct `Tweets.csv`. Error: {e}")

# --- MAIN APP ROUTER ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sentiment Predictor", "Data & Model Insights"])

if page == "Sentiment Predictor":
    page_predictor()
elif page == "Data & Model Insights":
    page_insights()

st.sidebar.info("App created by Tanishka.")
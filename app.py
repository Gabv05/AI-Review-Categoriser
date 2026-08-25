import streamlit as st
import pandas as pd
import plotly.express as px
from google_play_scraper import reviews, Sort
from transformers import pipeline

# page configuration
st.set_page_config(page_title="AI Feedback Categorizer", layout="wide")
st.title("📊 Product Feedback Categorizer")
st.markdown("Scrape live google store app reviews and categorize them using a local NLP model.")

# loading the AI Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    # smaller, faster model (DistilBERT) for quick CPU inference
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

classifier = load_model()

# defined categories 
categories = ["Bug Report", "Feature Request", "Pricing Complaint", "UX/UI Issue", "General Praise", "General Complaint"]

# Initialize session state to hold our dataframe across page reruns
if 'df' not in st.session_state:
    st.session_state.df = None

# user input 
app_id = st.text_input("Enter Google Play App ID (e.g., com.spotify.music):", "com.spotify.music")
review_count = st.slider("Number of reviews to process (higher number = higher CPU load):", 5, 50, 15)

if st.button("Scrape & Analyze"):
    with st.spinner("Scraping reviews and running AI inference..."):
        # scraping reviews
        try:
            result, _ = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=review_count
            )
            
            # extracting just the review text
            review_texts = [rev['content'] for rev in result]
            
        except Exception as e:
            st.error(f"Error scraping app: {e}")
            st.stop()

        # AI classification
        categorized_data = []
        for text in review_texts:
            # AI reads the text and assigns the best matching label from custom list
            ai_result = classifier(text, categories)
            best_label = ai_result['labels'][0]
            confidence = ai_result['scores'][0]
            
            categorized_data.append({
                "Review": text,
                "Category": best_label,
                "Confidence": f"{confidence:.2%}"
            })

        # Save the structured data into Streamlit's memory
        st.session_state.df = pd.DataFrame(categorized_data)


# --- RENDER DASHBOARD (Outside the button click) ---
# If we have data in our memory, draw the UI!
if st.session_state.df is not None:
    df = st.session_state.df
    
    # dashboard rendering
    st.success("Analysis Complete!")
    
    # create a layout with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Category Breakdown")
        # count the occurrences of each category
        category_counts = df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # draw an interactive Pie Chart using Plotly
        fig = px.pie(category_counts, names='Category', values='Count', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Raw AI Output")
        # display the interactive dataframe
        st.dataframe(df, use_container_width=True)

    # Note: No need for a column here if we just want it full width underneath
    st.divider()
    st.subheader("Export Data")

    # converting the DataFrame to a CSV format
    csv_data = df.to_csv(index=False).encode('utf-8')

    # download button
    st.download_button(
        label="📥 Download as CSV (Excel / Google Sheets export)",
        data=csv_data,
        file_name='categorized_feedback.csv',
        mime='text/csv',
    )

# CSS injecion
st.markdown("""
<style>
    /* 1. Import Futuristic Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Share+Tech+Mono&display=swap');

    /* 2. Base Theme Variables */
    :root {
        --bg-base: #0a0a12;
        --cyan-neon: #00f3ff;
        --magenta-neon: #ff00ff;
        --violet-neon: #8a2be2;
        --amber-accent: #ffb000;
        --text-main: #e0e6ed;
    }

    /* 3. Global App Background (Grid & Scanlines) */
    .stApp {
        background-color: var(--bg-base);
        /* Cyberpunk Grid Pattern */
        background-image: 
            linear-gradient(rgba(0, 243, 255, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 243, 255, 0.05) 1px, transparent 1px);
        background-size: 40px 40px;
        font-family: 'Share Tech Mono', monospace;
        color: var(--text-main);
    }

    /* Subtle CRT Scanline Overlay */
    .stApp::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.15) 50%);
        background-size: 100% 4px;
        z-index: 9999;
        pointer-events: none;
    }

    /* 4. Typography styling */
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        color: var(--cyan-neon) !important;
        text-shadow: 0 0 4px rgba(0, 243, 255, 0.4), 0 0 10px rgba(0, 243, 255, 0.2);
        letter-spacing: 2px;
    }

    p, span, div {
        font-family: 'Share Tech Mono', monospace;
    }

    /* 5. Terminal-Inspired Inputs */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div, 
    .stNumberInput > div > div > input {
        background-color: rgba(10, 10, 18, 0.8) !important;
        border: 1px solid var(--violet-neon) !important;
        color: var(--cyan-neon) !important;
        border-radius: 0px !important;
        box-shadow: inset 0 0 8px rgba(138, 43, 226, 0.3);
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border: 1px solid var(--cyan-neon) !important;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.5), inset 0 0 8px rgba(0, 243, 255, 0.3);
    }

    /* 6. Buttons (Neon Hover Effects) */
    .stButton > button, .stDownloadButton > button {
        background-color: transparent !important;
        color: var(--magenta-neon) !important;
        border: 2px solid var(--magenta-neon) !important;
        border-radius: 0px !important; /* Sharp sci-fi edges */
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(255, 0, 255, 0.2), inset 0 0 5px rgba(255, 0, 255, 0.2) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: var(--magenta-neon) !important;
        color: #000 !important;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.8), inset 0 0 10px rgba(255, 0, 255, 0.5) !important;
        transform: translateY(-2px);
    }

    /* 7. Dataframes & Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--cyan-neon);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
        background-color: rgba(0, 0, 0, 0.6);
    }
    
    /* 8. Restrained Glitch Effect on Title Hover */
    @keyframes glitch {
        0% { text-shadow: 2px 0 var(--magenta-neon), -2px 0 var(--cyan-neon); }
        50% { text-shadow: -2px 0 var(--magenta-neon), 2px 0 var(--cyan-neon); }
        100% { text-shadow: 2px 0 var(--magenta-neon), -2px 0 var(--cyan-neon); }
    }
    h1:hover {
        animation: glitch 0.3s linear infinite;
    }

    /* Hide Streamlit Branding for a standalone app feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .sentiment-neutral {
        color: #f39c12;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">😊 Sentiment Analysis App</h1>', unsafe_allow_html=True)
st.markdown("### Powered by Fine-tuned BERT Model")

# Initialize the model
@st.cache_resource
def load_model():
    """Load the sentiment analysis model from Hugging Face"""
    try:
        # Using pipeline for simplicity
        classifier = pipeline(
            "sentiment-analysis",
            model="mustehsannisarrao/fine-tune-bert-sentimental-analysis",
            tokenizer="mustehsannisarrao/fine-tune-bert-sentimental-analysis"
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
with st.spinner("Loading sentiment analysis model..."):
    classifier = load_model()

if classifier is None:
    st.error("Failed to load the model. Please check if the model name is correct.")
    st.stop()

# Sidebar for additional options
st.sidebar.title("Settings")
st.sidebar.markdown("### Model Information")
st.sidebar.write("**Model:** Fine-tuned BERT")
st.sidebar.write("**Owner:** mustehsannisarrao")
st.sidebar.write("**Task:** Sentiment Analysis")

st.sidebar.markdown("### About")
st.sidebar.info(
    "This app uses a fine-tuned BERT model to analyze the sentiment of text. "
    "It can classify text as positive, negative, or neutral."
)

# Main content area
tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch Analysis", "API Usage"])

with tab1:
    st.header("Analyze Single Text")
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type your text here...",
        height=100
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing sentiment..."):
                # Add a small delay to show the spinner
                time.sleep(0.5)
                
                # Perform sentiment analysis
                try:
                    result = classifier(text_input)[0]
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Label", result['label'])
                    
                    with col2:
                        st.metric("Confidence Score", f"{result['score']:.4f}")
                    
                    with col3:
                        confidence_percent = result['score'] * 100
                        st.metric("Confidence", f"{confidence_percent:.1f}%")
                    
                    # Visualize confidence
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.barh(['Confidence'], [confidence_percent], color='skyblue')
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Confidence (%)')
                    ax.axvline(x=80, color='red', linestyle='--', alpha=0.7, label='High Confidence')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Color-coded sentiment display
                    sentiment = result['label'].upper()
                    if 'POSITIVE' in sentiment:
                        st.markdown(f'<p class="sentiment-positive">🎉 This text appears to be POSITIVE!</p>', unsafe_allow_html=True)
                    elif 'NEGATIVE' in sentiment:
                        st.markdown(f'<p class="sentiment-negative">😞 This text appears to be NEGATIVE!</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="sentiment-neutral">😐 This text appears to be NEUTRAL!</p>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Batch Text Analysis")
    
    # Multiple text inputs
    st.subheader("Enter multiple texts (one per line):")
    batch_text = st.text_area(
        "Batch texts:",
        placeholder="Enter each text on a new line...\n\nExample:\nI love this product!\nThis is terrible.\nIt's okay, I guess.",
        height=150
    )
    
    if st.button("Analyze Batch", type="primary"):
        if batch_text.strip():
            texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
            
            if texts:
                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    try:
                        results = classifier(texts)
                        
                        # Create results dataframe
                        results_data = []
                        for i, (text, result) in enumerate(zip(texts, results)):
                            results_data.append({
                                'Text': text[:50] + '...' if len(text) > 50 else text,
                                'Full Text': text,
                                'Sentiment': result['label'],
                                'Confidence': result['score'],
                                'Confidence %': f"{result['score'] * 100:.1f}%"
                            })
                        
                        df = pd.DataFrame(results_data)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Summary statistics
                        sentiment_counts = df['Sentiment'].value_counts()
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Texts", len(texts))
                        with col2:
                            st.metric("Unique Sentiments", len(sentiment_counts))
                        with col3:
                            avg_confidence = df['Confidence'].mean() * 100
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                        
                        # Display dataframe
                        st.dataframe(df[['Text', 'Sentiment', 'Confidence %']], use_container_width=True)
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during batch analysis: {e}")
            else:
                st.warning("Please enter at least one valid text to analyze.")
        else:
            st.warning("Please enter some texts to analyze.")

with tab3:
    st.header("API Usage Example")
    st.markdown("""
    ### How to use this model in your code:
    
    ```python
    from transformers import pipeline
    
    # Load the model
    classifier = pipeline(
        "sentiment-analysis",
        model="mustehsannisarrao/fine-tune-bert-sentimental-analysis"
    )
    
    # Analyze text
    result = classifier("I love this amazing product!")
    print(result)
    # Output: [{'label': 'POSITIVE', 'score': 0.998}]
    
    # Batch analysis
    texts = [
        "This is wonderful!",
        "I hate this.",
        "It's okay, nothing special."
    ]
    results = classifier(texts)
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']}, Confidence: {result['score']:.4f}")
        print("---")
    ```
    
    ### Model Information:
    - **Model Name:** `mustehsannisarrao/fine-tune-bert-sentimental-analysis`
    - **Task:** Sentiment Analysis
    - **Framework:** Transformers
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Model Repository:** [mustehsannisarrao/fine-tune-bert-sentimental-analysis]"
    "(https://huggingface.co/mustehsannisarrao/fine-tune-bert-sentimental-analysis) | "
    "Built with ❤️ using Streamlit & Hugging Face"
)

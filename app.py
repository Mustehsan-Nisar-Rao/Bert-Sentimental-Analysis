import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .positive { 
        background-color: #d4edda; 
        color: #155724; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #28a745;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .negative { 
        background-color: #f8d7da; 
        color: #721c24; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #dc3545;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .neutral { 
        background-color: #fff3cd; 
        color: #856404; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #ffc107;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model with correct label mapping"""
    try:
        model_name = "mustehsannisarrao/fine-tune-bert-sentimental-analysis"
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create classifier
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )
        
        return classifier
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def map_label_to_sentiment(label):
    """Convert LABEL_X to actual sentiment based on your model's training"""
    label_mapping = {
        'LABEL_0': 'NEGATIVE',    # Based on: terrible, awful, worst
        'LABEL_1': 'NEUTRAL',     # Assumed based on standard 3-class sentiment
        'LABEL_2': 'POSITIVE'     # Based on: love, perfect
    }
    return label_mapping.get(label, label)

def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment"""
    emoji_map = {
        'POSITIVE': '🎉',
        'NEGATIVE': '😞', 
        'NEUTRAL': '😐'
    }
    return emoji_map.get(sentiment, '❓')

def create_confidence_chart(confidence, sentiment):
    """Create a confidence visualization"""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Color based on sentiment
    colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#f39c12'}
    color = colors.get(sentiment, '#95a5a6')
    
    ax.barh(['Confidence'], [confidence * 100], color=color, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)')
    ax.axvline(x=80, color='red', linestyle='--', alpha=0.3, label='High Confidence Threshold')
    ax.legend()
    
    # Add value text
    ax.text(confidence * 100, 0, f'{confidence*100:.1f}%', 
            va='center', ha='right', fontweight='bold')
    
    return fig

def main():
    st.markdown('<h1 class="main-header">😊 BERT Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Using your fine-tuned model: `mustehsannisarrao/fine-tune-bert-sentimental-analysis`")
    
    # Load model
    with st.spinner("Loading your fine-tuned BERT model..."):
        classifier = load_sentiment_model()
    
    if classifier is None:
        st.error("Failed to load the model. Please check your internet connection.")
        return
    
    # Sidebar with model info
    st.sidebar.title("Model Information")
    st.sidebar.markdown("""
    **Model Details:**
    - **Name:** Fine-tuned BERT
    - **Task:** Sentiment Analysis
    - **Classes:** 3 (Positive, Negative, Neutral)
    - **Label Mapping:**
      - LABEL_0 → NEGATIVE
      - LABEL_1 → NEUTRAL  
      - LABEL_2 → POSITIVE
    """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Single Analysis", "Batch Analysis", "Test Examples"])
    
    with tab1:
        st.header("Analyze Single Text")
        
        # Text input
        text_input = st.text_area(
            "Enter your text:",
            placeholder="Type your review, comment, or any text here...",
            height=120,
            key="single_input"
        )
        
        # Quick examples
        st.write("**Quick examples:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("😊 Positive Example", use_container_width=True):
                st.session_state.single_text = "I absolutely love this product! It's amazing and perfect!"
        with col2:
            if st.button("😞 Negative Example", use_container_width=True):
                st.session_state.single_text = "This is terrible and awful. Worst experience ever."
        with col3:
            if st.button("😐 Neutral Example", use_container_width=True):
                st.session_state.single_text = "It's okay, nothing special about this product."
        
        # Use session state if set
        if hasattr(st.session_state, 'single_text'):
            text_input = st.session_state.single_text
        
        # Analyze button
        if st.button("Analyze Sentiment", type="primary", key="analyze_single"):
            if text_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    try:
                        # Get prediction
                        result = classifier(text_input)[0]
                        original_label = result['label']
                        confidence = result['score']
                        
                        # Map to actual sentiment
                        sentiment = map_label_to_sentiment(original_label)
                        emoji = get_sentiment_emoji(sentiment)
                        
                        # Display results
                        st.subheader("Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Sentiment", f"{emoji} {sentiment}")
                        
                        with col2:
                            st.metric("Confidence Score", f"{confidence:.4f}")
                        
                        with col3:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Confidence chart
                        st.pyplot(create_confidence_chart(confidence, sentiment))
                        
                        # Color-coded result
                        if sentiment == 'POSITIVE':
                            st.markdown(f'<div class="positive">{emoji} POSITIVE sentiment detected! The text expresses positive emotion.</div>', unsafe_allow_html=True)
                        elif sentiment == 'NEGATIVE':
                            st.markdown(f'<div class="negative">{emoji} NEGATIVE sentiment detected! The text expresses negative emotion.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="neutral">{emoji} NEUTRAL sentiment detected! The text is neutral or mixed.</div>', unsafe_allow_html=True)
                        
                        # Debug info
                        with st.expander("Technical Details"):
                            st.write(f"**Raw model output:** {result}")
                            st.write(f"**Original label:** {original_label}")
                            st.write(f"**Mapped sentiment:** {sentiment}")
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Analysis")
        
        st.subheader("Enter multiple texts (one per line):")
        batch_text = st.text_area(
            "Batch texts:",
            placeholder="Enter each text on a new line...\n\nExample:\nI love this product!\nThis is terrible.\nIt's okay, nothing special.",
            height=200,
            key="batch_input"
        )
        
        if st.button("Analyze Batch", type="primary"):
            if batch_text.strip():
                texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
                
                if texts:
                    with st.spinner(f"Analyzing {len(texts)} texts..."):
                        try:
                            results = classifier(texts)
                            
                            # Process results
                            analysis_data = []
                            for i, (text, result) in enumerate(zip(texts, results)):
                                sentiment = map_label_to_sentiment(result['label'])
                                analysis_data.append({
                                    'Text': text,
                                    'Original_Label': result['label'],
                                    'Sentiment': sentiment,
                                    'Confidence': result['score'],
                                    'Confidence_Percent': f"{result['score'] * 100:.1f}%"
                                })
                            
                            df = pd.DataFrame(analysis_data)
                            
                            # Display summary
                            st.subheader("Batch Analysis Summary")
                            
                            sentiment_counts = df['Sentiment'].value_counts()
                            total_texts = len(df)
                            avg_confidence = df['Confidence'].mean() * 100
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Texts", total_texts)
                            with col2:
                                st.metric("Positive", sentiment_counts.get('POSITIVE', 0))
                            with col3:
                                st.metric("Negative", sentiment_counts.get('NEGATIVE', 0))
                            with col4:
                                st.metric("Neutral", sentiment_counts.get('NEUTRAL', 0))
                            
                            # Display detailed results
                            st.subheader("Detailed Results")
                            st.dataframe(df[['Text', 'Sentiment', 'Confidence_Percent']], use_container_width=True)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
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
        st.header("Test Examples")
        st.markdown("Test your model with these predefined examples:")
        
        test_cases = [
            ("I love this product! It's absolutely amazing!", "Should be POSITIVE"),
            ("This is terrible and awful. Worst purchase ever.", "Should be NEGATIVE"),
            ("It's okay, nothing special about it.", "Should be NEUTRAL"),
            ("The product is perfect and excellent!", "Should be POSITIVE"),
            ("This is the worst experience I've ever had.", "Should be NEGATIVE"),
            ("It's fine, I have no strong feelings.", "Should be NEUTRAL")
        ]
        
        for i, (text, expected) in enumerate(test_cases):
            if st.button(f"Test: {text[:30]}...", key=f"test_{i}"):
                with st.spinner("Testing..."):
                    result = classifier(text)[0]
                    predicted_sentiment = map_label_to_sentiment(result['label'])
                    
                    st.write(f"**Text:** {text}")
                    st.write(f"**Expected:** {expected}")
                    st.write(f"**Predicted:** {predicted_sentiment} (confidence: {result['score']:.4f})")
                    
                    if expected.split()[-1].lower() == predicted_sentiment.lower():
                        st.success("✅ Prediction matches expectation!")
                    else:
                        st.error("❌ Prediction doesn't match expectation!")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Model:** [mustehsannisarrao/fine-tune-bert-sentimental-analysis]"
        "(https://huggingface.co/mustehsannisarrao/fine-tune-bert-sentimental-analysis) | "
        "Built with ❤️ using Streamlit & Hugging Face Transformers"
    )

if __name__ == "__main__":
    main()

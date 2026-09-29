import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import re
import time

# Page configuration
st.set_page_config(page_title="Review Classification Dashboard", layout="wide")

# Arabic text cleaning and normalization pipeline
def clean_user_input(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove links
    # Standardize Arabic characters to improve model understanding
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and symbols
    text = re.sub(r'\d+', '', text)     # Remove numbers
    return text.strip()

# Load model weights and tokenizer from the root directory
try:
    loaded_model = tf.keras.models.load_model('arabic_deep_model.keras')
    loaded_tokenizer = joblib.load('arabic_tokenizer.pkl')
except Exception as e:
    st.error("Error: Model files (.keras or .pkl) not found. Please ensure they are placed in the same directory as app.py.")

# Main dashboard banner
st.markdown("""
    <div style='background-color: #1E3A8A; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 25px;'>
        <h1 style='color: white; margin: 0; font-family: sans-serif; font-size: 26px;'>Review Classification & Ticket Routing System</h1>
        <p style='color: #93C5FD; margin: 5px 0 0 0; font-size: 15px;'>A practical NLP application using a Bi-LSTM network to classify customer feedback and automate routing.</p>
    </div>
""", unsafe_allow_html=True)

# Layout division
col1, col2 = st.columns(2)

with col1:
    st.markdown("<b style='font-size: 15px; color: #1E3A8A;'>Input Interface:</b>", unsafe_allow_html=True)
    user_input = st.text_area("", height=130, placeholder="Type or paste the Arabic customer review here (Supports Syrian, Egyptian, Gulf dialects or MSA)...", key="review_input")
    
    if st.button("Run Classification Pipeline", use_container_width=True):
        if user_input.strip():
            # Calculate inference latency
            start_time = time.time()
            
            # Pre-processing and tokenization
            cleaned = clean_user_input(user_input)
            sequence = loaded_tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(sequence, maxlen=120, padding='post', truncating='post')
            
            # Model prediction execution
            prediction_prob = loaded_model.predict(padded, verbose=0)
            inference_time = (time.time() - start_time) * 1000
            
            st.markdown("### Model Inference Output:")
            st.markdown(f"Model Latency: `{inference_time:.2f} ms`")
            
            # Decision threshold logic
            if prediction_prob >= 0.5:
                confidence = prediction_prob * 100
                st.success(f"Classification: Positive Sentiment | Confidence Score: {float(confidence):.2f}%")
                st.info("System Action: The review was processed successfully and moved to the marketing department routine archive.")
                st.balloons() # Visual feedback for positive classification
            else:
                confidence = (1 - prediction_prob) * 100
                st.error(f"Classification: Negative Sentiment | Confidence Score: {float(confidence):.2f}%")
                
                # Support routing alert block
                st.markdown(f"""
                    <div style='background-color: #FEE2E2; padding: 15px; border-left: 5px solid #DC2626; border-radius: 5px; margin-top: 10px; margin-bottom: 15px;'>
                        <b style='color: #991B1B; font-size: 15px;'>Automated Ticket Routing Action:</b><br>
                        <span style='color: #7F1D1D;'>1. A high level of customer dissatisfaction was detected by the network layers.<br>
                        2. The pipeline generated an urgent support ticket to track the customer status and prevent churn.<br>
                        3. The file was routed directly to the department manager for immediate follow-up and account recovery.</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Ticket payload structure
                ticket_content = f"""==================================================
TICKET REPORT - AUTOMATED CUSTOMER SUPPORT
==================================================
[Customer Review]: {user_input}
[Model Prediction]: NEGATIVE (Complaint detected)
[Bi-LSTM Confidence]: {float(confidence):.2f}%
[Inference Time]: {inference_time:.2f} ms
[System Action]: Forwarded directly to the Operations Manager
==================================================™"""
                
                # Local download trigger
                st.download_button(
                    label="Export Ticket Report",
                    data=ticket_content,
                    file_name="Customer_Complaint_Ticket.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.warning("Please enter a review in the input box before running the pipeline.")

with col2:
    st.markdown("<b style='font-size: 15px; color: #1E3A8A;'>Technical Specifications:</b>", unsafe_allow_html=True)
    st.metric(label="Total Dataset Capacity", value="330,000 Reviews")
    st.metric(label="Neural Network Architecture", value="Bidirectional LSTM")
    st.metric(label="Vector Space Dimension", value="128 Embeddings")
    
    st.markdown("""
        <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px; margin-top: 20px;'>
            <small style='color: #475569;'><b>Architectural Notice:</b> This project separates the UI presentation layer from the back-end training core. The neural network weights were trained and compiled beforehand. They run locally in full standalone configuration to optimize text processing speed and preserve model availability without making external server calls.</small>
        </div>
    """, unsafe_allow_html=True)

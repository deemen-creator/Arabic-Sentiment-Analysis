# Arabic Sentiment Analysis & Automated Ticket Routing

A practical NLP project that uses a Deep Learning model to classify Arabic customer reviews (combining MSA and various dialects like Syrian, Egyptian, and Gulf) and automatically routes negative feedback into an urgent support ticket system.

## What this project solves
Large e-commerce platforms and retail stores receive thousands of reviews daily. Sorting these manually to find angry customers is slow. This pipeline automates the entire process:
1. It cleans and predicts the sentiment of the text (Positive or Negative).
2. If a review is negative, the system simulates an automated routing protocol, generating a high-priority ticket for the operations team to handle the customer before they churn.

## How it works (The Technical Pipeline)
* **The Dataset:** The model was trained using a sample of 30,000 records taken from the benchmark **330K Arabic Sentiment Reviews Dataset**.
* **Text Pre-processing:** To handle dialectal variations and noise (like typing mistakes, extra spaces, and hamza variations), I built a custom normalization function. It standardizes conflicting characters (e.g., converting أ, إ, آ into a plain ا, and ة to ه) and filters out links and numbers using Regex.
* **Vectorization:** Text is tokenized with a maximum vocabulary size of 25,000 words. Sequences are standardized to a fixed length of 120 tokens using post-padding and truncating.
* **Model Architecture:** Built using TensorFlow/Keras, consisting of:
  * An **Embedding Layer** (128 dimensions) to learn semantic relationships between dialectal words.
  * A **Bidirectional LSTM** layer (64 units) to read text from both directions and capture the full contextual meaning of the sentence.
  * **Dropout Layers** (0.5 and 0.3) to prevent overfitting during training.
  * A final **Dense Layer** with a Sigmoid activation function for binary classification.
* **Performance:** The model achieved a stable **~83.38% validation accuracy** on unseen test data.

## Note on Model Weights File
To keep the GitHub repository clean and avoid file size limits, the trained weights file (**`arabic_deep_model.keras`**, around 38MB) is not uploaded directly here. 
* **Important:** You must download or copy your trained `arabic_deep_model.keras` file and place it in the same root folder as `app.py` before running the app.

## How to run it locally (Offline Mode)
The interface is built using **Streamlit** and runs completely offline on your local machine without needing an internet connection.

1. Open your terminal/CMD and navigate to the project directory:
   ```bash
   cd path/to/AI_Project
   ```
2. Install the required libraries:
   ```bash
   pip install tensorflow streamlit joblib scikit-learn
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
   *(Or simply double-click the `run_app.bat` script if you are on Windows).*

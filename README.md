# Arabic Sentiment Analysis & Automated Ticket Routing

A practical NLP project that uses an optimized linear classification pipeline to classify Arabic customer reviews (combining MSA and various dialects like Syrian, Egyptian, and Gulf) and automatically routes negative feedback into an urgent support ticket system.

## What this project solves
Large e-commerce platforms and retail stores receive thousands of reviews daily. Sorting these manually to find angry customers is slow. This pipeline automates the entire process:
1. It cleans and predicts the sentiment of the text (Positive or Negative).
2. If a review is negative, the system simulates an automated routing protocol, generating a high-priority ticket for the operations team to handle the customer before they churn.

## How it works (The Technical Pipeline)
* **The Dataset:** The model was trained using a sample of 50,000 records taken from the benchmark **330K Arabic Sentiment Reviews Dataset**.
* **Text Pre-processing:** To handle dialectal variations and noise (like typing mistakes and hamza variations), I built a custom normalization function. It standardizes conflicting characters (e.g., converting أ, إ, آ into a plain ا, and ة to ه) and compresses structural character repetitions (stretching) using Regex.
* **Feature Engineering (The Secret Weapon):** Instead of treating words as isolated blocks, this pipeline extracts a dense matrix of 50,000 parallel features:
  * **Word-level TF-IDF (15,000 features):** Captures broad phrases and global context via single words, word-pairs, and triplets.
  * **Character-level TF-IDF (35,000 features):** Slices strings into sub-word letter blocks (2 to 5 characters) to robustly capture core sentiment despite spelling mistakes or slang suffixes.
* **Model Configuration:** Configured a high-performance **Logistic Regression** framework optimized via a **SAGA Solver** with balanced class scaling parameters to manage dialect distributions cleanly.
* **Performance:** The optimized system achieved a stable **89.08% clean evaluation accuracy** on unseen verification data.

## Note on Production Binaries
To keep the GitHub repository clean and avoid file size limits, the trained deployment weights are saved as standalone binaries:
* **`arabic_deep_model.keras`** (The compiled model coefficients)
* **`arabic_tokenizer.pkl`** (The feature union map)
* **Important:** You must place both your generated files inside the root directory alongside `app.py` before initiating execution.

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

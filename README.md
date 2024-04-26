## SignScribe: Sign Language Interpretation and Speech-to-Sign App

SignScribe is a web application built with Streamlit and Python that bridges the gap between sign language and spoken language communication. We plan that it utilizes three machine learning models to achieve seamless communication:

* **Sign Language Interpretation:** This model interprets hand gestures captured through a webcam and converts them into corresponding words.
* **Sentence Completion/Correction:** The interpreted words are then processed by another model, which constructs grammatically correct and complete sentences from them.
* **Speech-to-Sign Conversion:** Finally, the completed sentences are fed into a third model, which generates real-time sign language animations for the other user to understand.

**Getting Started**

To run SignScribe, you'll need the following:

* Python 3.x (with required libraries: Streamlit, OpenCV, and appropriate machine learning frameworks for your chosen models)
* A webcam

1. Clone this repository:

   ```bash
   git clone https://github.com/revv2703/signscribe.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python run main.py
   ```
import gradio as gr
from transformers import pipeline
import time
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize Hugging Face API key
API_KEY = "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"

# Set up Hugging Face model pipelines

sentiment_pipeline = pipeline("sentiment-analysis", model="rohan11129/my_finetuned_sentiment_model", token=API_KEY)
qa_pipeline = pipeline("question-answering", model="rohan11129/my_finetuned_sentiment_model", token=API_KEY)
translation_pipeline = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi", token=API_KEY)
translation_pipeline_marathi = pipeline("translation_en_to_mr", model="Helsinki-NLP/opus-mt-en-mr", token=API_KEY)
speech_recognition_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", token=API_KEY)
classification_pipeline = pipeline("zero-shot-classification", model="rohan11129/my_finetuned_sentiment_model", token=API_KEY)

# Function to analyze sentiment of customer query
def analyze_sentiment(query):
    sentiment = sentiment_pipeline(query)[0]
    return f"Sentiment: {sentiment['label']}, Confidence: {sentiment['score']:.2f}"

# Function to classify the customer query
def classify_query(query):
    labels = ["Product Inquiry", "Order Status", "Returns", "Payment Issue", "General Inquiry"]
    classification = classification_pipeline(query, candidate_labels=labels)
    top_class = classification["labels"][0]
    top_score = classification["scores"][0]
    return f"Category: {top_class}, Confidence: {top_score:.2f}"

# Function to answer customer queries about products
def answer_question(question,context):
   # context = "This is a demo e-commerce platform where we offer a wide range of products including electronics, clothing, and home appliances. Please ask about product features, reviews, and policies."
    response = qa_pipeline(question=question,context=context)
    return response['answer']

# Function to translate response to Hindi and Marathi
def translate_response(response, language="Hindi"):
    if language == "Hindi":
        translated_text = translation_pipeline(response)[0]["translation_text"]
    else:
       translated_text = translation_pipeline_marathi(response)[0]["translation_text"]
    return translated_text

# Function to transcribe audio queries
def speech_to_text(audio):
    transcription = speech_recognition_pipeline(audio)["text"]
    return transcription

# Metrics tracking dictionary
metrics = {
    "sentiment_accuracy": [],
    "classification_accuracy": [],
    "qa_accuracy": [],
    "translation_time": [],
    "response_time": [],
    "speech_recognition_accuracy": []
}

def record_metric(metric_name, value):
    metrics[metric_name].append(value)

# Wrapper function for complete pipeline
def customer_service_pipeline(audio=None, context=None,text_query=None, language="Hindi"):
    start_time = time.time()

    if audio:
        # Speech recognition
        transcription = speech_to_text(audio)
        record_metric("speech_recognition_accuracy", random.uniform(0.8, 1.0))  # Mock metric
    else:
        transcription = text_query

    # Sentiment analysis
    sentiment = analyze_sentiment(transcription)
    record_metric("sentiment_accuracy", random.uniform(0.85, 1.0))  # Mock metric

    # Classification
    classification = classify_query(transcription)
    record_metric("classification_accuracy", random.uniform(0.85, 1.0))  # Mock metric

    # Question answering
    answer = answer_question(transcription,context)
    record_metric("qa_accuracy", random.uniform(0.9, 1.0))  # Mock metric

    # Translation
    translation_start = time.time()
    translated_answer = translate_response(answer, language)
    translation_time = time.time() - translation_start
    record_metric("translation_time", translation_time)

    response_time = time.time() - start_time
    record_metric("response_time", response_time)

    # Display metrics summary
    metrics_summary = {
        "Average Sentiment Accuracy": f"{sum(metrics['sentiment_accuracy']) / len(metrics['sentiment_accuracy']):.2f}" if metrics["sentiment_accuracy"] else "N/A",
        "Average Classification Accuracy": f"{sum(metrics['classification_accuracy']) / len(metrics['classification_accuracy']):.2f}" if metrics["classification_accuracy"] else "N/A",
        "Average QA Accuracy": f"{sum(metrics['qa_accuracy']) / len(metrics['qa_accuracy']):.2f}" if metrics["qa_accuracy"] else "N/A",
        "Average Translation Time (s)": f"{sum(metrics['translation_time']) / len(metrics['translation_time']):.2f}" if metrics["translation_time"] else "N/A",
        "Average Response Time (s)": f"{sum(metrics['response_time']) / len(metrics['response_time']):.2f}" if metrics["response_time"] else "N/A",
        "Average Speech Recognition Accuracy": f"{sum(metrics['speech_recognition_accuracy']) / len(metrics['speech_recognition_accuracy']):.2f}" if metrics["speech_recognition_accuracy"] else "N/A"
    }

    return transcription, sentiment, classification, answer, translated_answer, metrics_summary

# Gradio interface with custom CSS
css = """
body {
    background-color: #f0fff8; /* Soft light gray background */
    font-family: Arial, sans-serif;
    color: #333;
    margin: 0;
    padding: 0;
}

.gradio-container {
    max-width: 800px;
    margin: 20px auto;
    padding: 20px;
    background-color: #4b0082; /* White container for content */
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
}

h1 {
    font-size: 30px;
    color: #ffffff; /* Deep purple for headings */
    text-align: center;
    margin-bottom: 22px;
}

label {
    font-size:20px;
    color: #999;
    font-weight: bold; /* Make labels bold */
}

textarea {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
    width: 100%;
    color: #333;
    background-color: #fafafa;
}

.gradio-block button {
    background-color: #6a08sd; /* Deep purple button */
    color: white;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 15px;
    font-weight: bold;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.gradio-block button:hover {
    background-color: #5c0dab; /* Slightly darker on hover */
}

.gradio-block button:focus {
    outline: none;
    box-shadow: 0px 0px 4px #5c0dab;
}

.gradio-block input[type='radio'] {
    margin: 10px 5px;
}

.gradio-block .output {
    background-color: #f5f5f5; /* Light gray output background */
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 12px;
    margin-top: 12px;
}

.gradio-block .output p {
    font-size: 16px;
    color: #333;
}

.gradio-block label {
    font-size: 14px;
    color: #333;
    font-weight: bold;
}
"""


# Gradio interface with submit button
interface = gr.Interface(
    fn=customer_service_pipeline,
    inputs=[
        gr.Audio(type="filepath", label="Ask by Speech"),
        gr.Textbox(label="About Product", placeholder="Provide context for the question"),
        gr.Textbox(label="Customer Query", placeholder="Type your question here"),
        gr.Radio(["Hindi", "Marathi"], label="Translate to", value="Hindi")
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Sentiment Analysis"),
        gr.Textbox(label="Classification"),
        gr.Textbox(label="Answer to Query"),
        gr.Textbox(label="Translated Answer"),
        gr.JSON(label="Metrics Summary")
    ],
    live=False,  # Set to False to enable the submit button
    title="E-commerce Customer Service Assistant",
    css=css,
    allow_flagging="never"  # Prevent flagging for simplicity
)

interface.launch(share=True)


# Imports
#
# 3rd-party modules
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk import sent_tokenize
from collections import Counter


# Download NLTK data
nltk.download('punkt')

# Load a pre-trained model and tokenizer for summarization
model_name = "facebook/bart-large-cnn"  # You can also use "t5-base" or another model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


# Step 1: Specific Event Extraction
def extract_events(text):
    """
    Extract events from the input text.
    Here, sentences are considered as events.
    """
    sentences = sent_tokenize(text)
    return sentences


# Step 2: Event Abstraction and Generalization
def abstract_events(events):
    """
    Generalize the extracted events using a summarization model.
    Each event (sentence) is abstracted and summarized.
    """
    abstracted_events = [summarizer(event, max_length=30, min_length=10, do_sample=False)[0]['summary_text'] for event
                         in events]
    return abstracted_events


# Step 3: Common Event Statistics
def common_events(abstracted_events):
    """
    Analyze the abstracted events to find out which events are most common.
    """
    event_counter = Counter(abstracted_events)
    # Select the most common events (those that appear more than once)
    common_events = [event for event, count in event_counter.items() if count > 1]
    return common_events


# Step 4: Summary Generation
def generate_summary(common_events):
    """
    Generate a concise summary from the most common events.
    """
    combined_text = " ".join(common_events)
    summary = summarizer(combined_text, max_length=100, min_length=50, do_sample=False)[0]['summary_text']
    return summary


# Chain-of-Event Prompting Process
def chain_of_event_prompting(texts):
    """
    Full Chain-of-Event Prompting workflow:
    1. Extract events from multiple texts.
    2. Generalize and abstract the events.
    3. Analyze the commonality of the events.
    4. Generate a summary from the common events.
    """
    all_events = []
    for text in texts:
        events = extract_events(text)
        abstracted_events = abstract_events(events)
        all_events.extend(abstracted_events)

    common_events_list = common_events(all_events)
    summary = generate_summary(common_events_list)

    return summary


# Example Usage
if __name__ == "__main__":
    # Example input texts
    texts = [
        "The company announced a new product line which will be launched next month.",
        "A new product line is being developed by the company, with a launch expected in the near future.",
        "Next month, the company plans to introduce a new series of products to the market."
    ]

    # Perform Chain-of-Event Prompting
    final_summary = chain_of_event_prompting(texts)
    print("Final Summary:", final_summary)

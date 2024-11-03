import pandas as pd
from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ast import literal_eval
from huggingface_hub import login

# Define the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load your dataset
data = pd.read_json('ecommerce_data_modified.jsonl', lines=True)



# Mapping string labels to integers
def label_to_int(label):
    if label == "product_inquiry":
        return 1
    elif label == "complaint":
        return 0
    elif label == "return_request":
        return 1
    elif label == "payment_issue":
        return 0
    elif label == "feedback":
        return 1
    elif label == "order_status":
        return 1
    else:
        return ValueError(f"Unexpected label: {label}")

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(data)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['customer_query'], padding="max_length", truncation=True)

# Map labels to integers
def preprocess_data(examples):
    examples['label'] = [label_to_int(label) for label in examples['label']]
    return examples

# Tokenize and preprocess the dataset
dataset = dataset.map(preprocess_data, batched=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Splitting the dataset into train and test
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']


# Define training arguments
training_args = TrainingArguments(
    output_dir='./my_finetuned_sentiment_model',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model('./my_finetuned_sentiment_model')
login("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")
trainer.push_to_hub("rohan11129/my_finetuned_sentiment_model")
model.push_to_hub("rohan11129/my_finetuned_sentiment_model")
tokenizer.push_to_hub("rohan11129/my_finetuned_sentiment_model")
#model.config.push_to_hub("rohan11129/my_finetuned_sentiment_model")

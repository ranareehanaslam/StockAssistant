import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_DISABLED'] = "true"
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification
)
from datasets import Dataset
 
#######################################
########## FinBERT training ###########
#######################################

class args:
    model = 'ProsusAI/finbert'

df = pd.read_csv('all-data.csv', 
                 names = ['labels','messages'],
                 encoding='ISO-8859-1')

df = df[['messages', 'labels']]

le = LabelEncoder()
df['labels'] = le.fit_transform(df['labels'])

X, y = df['messages'].values, df['labels'].values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1) 
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2) 

train_dataset_raw = Dataset.from_dict({'text':xtrain, 'labels':ytrain})
valid_dataset_raw = Dataset.from_dict({'text':xvalid, 'labels':yvalid})

tokenizer = AutoTokenizer.from_pretrained(args.model)

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True)

train_dataset = train_dataset_raw.map(tokenize_fn, batched=True)
valid_dataset = valid_dataset_raw.map(tokenize_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(args.model)

train_args = TrainingArguments(
    './Finbert Trained/',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=2*16,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,    
    do_eval=True,
    do_train=True,
    do_predict=True,
    evaluation_strategy='epoch',
    save_strategy="no",
)

trainer = Trainer(
    model,
    train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer 
)

trainer.train()

# saving the model and the weights
model.save_pretrained('fine_tuned_FinBERT')
# saving the tokenizer
tokenizer.save_pretrained("fine_tuned_FinBERT/tokenizer/")


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch.nn.functional as F
import numpy as np
import re
import textstat

class RobertaWithFeatures(RobertaForSequenceClassification):
    def __init__(self, config, num_features):
        super().__init__(config)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(config.hidden_size + num_features, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 1)  # Final output layer
        
        # Dropout layer for regularization
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask=None, features=None):
        # Step 1: Pass input through RoBERTa
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        roberta_output = outputs.last_hidden_state[:, 0]  # [CLS] token output
        
        # Step 2: Concatenate RoBERTa output with additional features
        combined_input = torch.cat((roberta_output, features), dim=1)
        
        # Step 3: Pass through fully connected layers with ReLU and Dropout
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        x = self.dropout(F.relu(self.fc3(x)))
        
        # Step 4: Final prediction
        scalar_output = self.fc4(x)
        
        return scalar_output

def textstat_features(text):
    linsear_write_formula= textstat.linsear_write_formula(text)
    text_standard= textstat.text_standard(text, float_output=True)

    return features

def sentence_count(text):
    sentence = 0
    for i in text:
        if i in ".?!":
            sentence += 1
    return sentence

def count_paragraphs(text):
    # Split the text into paragraphs using newline characters
    paragraphs = text.split('\n')
    
    # Filter out any empty paragraphs that may result from the split
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    
    # Count the number of non-empty paragraphs
    num_paragraphs = len(paragraphs)
    
    return num_paragraphs

def average_paragraph_length(text):
    # Split the text into paragraphs using newline characters
    paragraphs = text.split('\n')
    
    # Filter out any empty paragraphs that may result from the split
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    
    # Calculate the total number of sentences in all paragraphs
    total_sentences = sum(sentence_count(paragraph) for paragraph in paragraphs)
    
    # Calculate the average paragraph length
    average_length = total_sentences / len(paragraphs) if paragraphs else 0
    
    return average_length

def extract_features(text):
    words = text.split()
    num_words = len(words)
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
    para_count = count_paragraphs(text)
    average_para_length = average_paragraph_length(text)
    linsear_write_formula= textstat.linsear_write_formula(text)
    text_standard= textstat.text_standard(text, float_output=True)
    return [num_words, avg_word_length, para_count, average_para_length, linsear_write_formula, text_standard]

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    return text    

cList = {
    "ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",  "could've": "could have",
    "couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not",
    "hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
    "he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is",
    "I'd": "I would","I'd've": "I would have","I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have","isn't": "is not",
    "it'd": "it had","it'd've": "it would have","it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam",
    "mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not",
    "mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
    "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she would",
    "she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have",
    "shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have",
    "that's": "that is","there'd": "there had","there'd've": "there would have","there's": "there is","they'd": "they would",
    "they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have",
    "to've": "to have","wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
    "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
    "what's": "what is","what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is",
    "where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is",
    "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
    "wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would","y'all'd've": "you all would have",
    "y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have","you'll": "you will",
    "you'll've": "you will have","you're": "you are","you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    """
    Expand contractions in the given text based on the cList dictionary.
    
    Parameters:
    text (str): The input text containing contractions.
    c_re (re.Pattern): The compiled regex pattern for matching contractions.
    
    Returns:
    str: The text with contractions expanded.
    """
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def removeHTML(x):
    """
    Remove HTML tags from the given text.
    
    Parameters:
    x (str): The input text containing HTML tags.
    
    Returns:
    str: The text with HTML tags removed.
    """
    html = re.compile(r'<.*?>')
    return html.sub(r'', x)


def dataPreprocessing(x):
    """
    Preprocess the input text by performing a series of cleaning steps:
    - Convert to lowercase
    - Remove HTML tags
    - Remove mentions
    - Remove contractions
    - Remove digits
    - Remove URLs
    - Remove extra whitespaces
    - Expand contractions
    - Remove repeated punctuation
    
    Parameters:
    x (str): The input text to preprocess.
    
    Returns:
    str: The cleaned and preprocessed text.
    """
    x = x.lower()  # Convert to lowercase
    x = removeHTML(x)  # Remove HTML tags
    x = re.sub("@\w+", '', x)  # Remove mentions
    x = re.sub("'\d+", '', x)  # Remove contractions
    x = re.sub("\d+", '', x)  # Remove digits
    x = re.sub("http\w+", '', x)  # Remove URLs
    x = re.sub(r"\s+", " ", x)  # Remove extra whitespaces
    x = expandContractions(x)  # Expand contractions
    x = re.sub(r"\.+", ".", x)  # Remove repeated periods
    x = re.sub(r"\,+", ",", x)  # Remove repeated commas
    x = x.strip()  # Remove leading and trailing whitespaces
    x = clean_text(x)
    return x
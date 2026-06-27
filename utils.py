
import re
import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    """
    Clean and normalise a raw SMS message.
    Used by both the training notebook (Member 2) and the Streamlit app (Member 3).
    DO NOT modify this function — any change will break model predictions.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+',           ' ', text)
    text = re.sub(r'#\w+',           ' ', text)
    text = re.sub(r'<.*?>',          ' ', text)
    text = re.sub(r'\d+',            ' ', text)
    text = re.sub(r'[^\w\s]',        ' ', text)
    text = re.sub(r'[^a-z\s]',       ' ', text)
    text = re.sub(r'\s+',            ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

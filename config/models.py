import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))
NLP = spacy.load("en_core_web_sm")

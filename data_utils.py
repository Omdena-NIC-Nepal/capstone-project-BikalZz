import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from spacy import displacy
from collections import defaultdict
from textblob import TextBlob
import ast, subprocess, importlib.util


# Load spacy model from NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def load_data():
    """Load and preprocess the climate data."""

    # Load the data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'data', 'processed_data', 'combined_data.csv')
    data = pd.read_csv(data_file)

    # Basic preprocessings
    data['year'] = data['year'].astype(int)

    return data

def preprocess_data(data, target_col = 'avg_max_temp', test_size=0.2, random_state=42):
    """Preprocess data for modeling."""

    # Seperate features and target
    X = data.drop(columns = [target_col])
    y = data[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state
    )

    # Scale numerical features
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=np.number).columns
    print(num_cols)
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler

def save_model(model, model_name):
    """Save trained model to disk."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'models', model_name)
    joblib.dump(model, f"{data_file}.pkl")

def load_model(model_name):
    """Load trained model from disk."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'models', model_name)
    return joblib.load(f"{data_file}.pkl")

def analyze_text(text):
    """Perform NLP analysis on text using spacy."""
    doc = nlp(text)
    return {
        "tokens": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "pos_tags": [(token.text, token.pos_) for token in doc],
        "entities": [(ent.text, ent.label_) for ent in doc.ents]
    }

def analyze_climate_text(text):
    """
    Specialized climate text analysis that extends basic analyze_text()
    with climate-specific insights
    """
    doc = nlp(text)
    
    # Climate-specific entity recognition
    climate_ents = []
    for ent in doc.ents:
        if ent.label_ in ["DATE", "LOC", "ORG", "GPE"]:  # Geographic and temporal references
            climate_ents.append((ent.text, ent.label_))
        elif ent.label_ == "QUANTITY" and any(t in ent.text for t in ["°C", "°F", "mm"]):
            climate_ents.append((ent.text, "CLIMATE_MEASUREMENT"))
    
    # Temperature mentions with context
    temp_mentions = []
    for sent in doc.sents:
        if any(t.lemma_ in ["temperature", "temp", "warm", "cold", "heat", "cool"] for t in sent):
            temp_mentions.append(sent.text)
    
    # Trend indicators
    trends = []
    trend_words = {
        "increase": "UP", "rise": "UP", "grow": "UP", "higher": "UP",
        "decrease": "DOWN", "fall": "DOWN", "drop": "DOWN", "lower": "DOWN"
    }
    for token in doc:
        if token.lemma_ in trend_words:
            trends.append((trend_words[token.lemma_], token.text))
    
    # Climate term frequency
    climate_terms = defaultdict(int)
    for token in doc:
        if token.lemma_ in ["temperature", "precipitation", "humidity", "rain", "drought"]:
            climate_terms[token.lemma_] += 1
    
    return {
        # Basic NLP
        "tokens": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "pos_tags": [(token.text, token.pos_) for token in doc],
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        
        # Climate-specific
        "climate_entities": climate_ents,
        "temperature_mentions": temp_mentions,
        "trend_indicators": trends,
        "climate_term_frequency": dict(climate_terms),
        "visualization": displacy.render(doc, style="ent", page=True)
    }
    
def load_lexicons():
    """Load only negative and positive lexicons"""
    try:
        base_path = os.path.join(os.path.dirname(__file__), "data", "sentiment_data")
        negative_df = pd.read_csv(os.path.join(base_path, "negative.csv"), usecols=['Word', 'sentiment', 'category'])
        positive_df = pd.read_csv(os.path.join(base_path, "positive.csv"), usecols=['Word', 'sentiment', 'category'])
        
    except FileNotFoundError as e:
        raise Exception(f"Lexicon file is missing.")

    lexicon = {}
    
    # Process negative words with enhanced climate weights
    for _, row in negative_df.iterrows():
        try:
            sentiment_dict = ast.literal_eval(row['sentiment'])
            lexicon[row['Word'].lower()] = {
                'polarity': -1 * abs(float(sentiment_dict['compound'])),  # Ensure negative
                'category': 'negative',
                'neg': min(1.0, float(sentiment_dict['neg']) * 1.2),  # Boost climate negativity
                'pos': 0,
                'neu': max(0, float(sentiment_dict['neu']) * 0.8)  # Reduce neutral weight
            }
        except:
            continue

    # Process positive words with climate context
    for _, row in positive_df.iterrows():
        try:
            sentiment_dict = ast.literal_eval(row['sentiment'])
            word = row['Word'].lower()
            
            # Climate-specific adjustments
            if word in ['warming']:  # Override positive warming
                lexicon[word] = {
                    'polarity': -0.7,
                    'category': 'negative',
                    'neg': 0.8,
                    'pos': 0.1,
                    'neu': 0.1
                }
            else:
                lexicon[word] = {
                    'polarity': abs(float(sentiment_dict['compound'])),
                    'category': 'positive',
                    'neg': 0,
                    'pos': min(1.0, float(sentiment_dict['pos']) * 1.1),  # Boost positivity
                    'neu': max(0, float(sentiment_dict['neu']) * 0.7)
                }
        except:
            continue

    # Climate term overrides
    climate_terms = {
        'climate': {'polarity': -0.3, 'category': 'negative', 'neg': 0.6, 'pos': 0.1, 'neu': 0.3},
        'change': {'polarity': -0.4, 'category': 'negative', 'neg': 0.7, 'pos': 0.0, 'neu': 0.3},
        'crisis': {'polarity': -0.9, 'category': 'negative', 'neg': 1.0, 'pos': 0.0, 'neu': 0.0}
    }
    
    for term, values in climate_terms.items():
        lexicon[term] = values

    return lexicon

CLIMATE_LEXICON = load_lexicons()

def analyze_climate_sentiment(text):
    """Sentiment analysis using only positive/negative lexicons"""
    blob = TextBlob(text.lower())
    
    # Initialize with default values
    result = {
        'score': 0,
        'label': 'neutral',
        'neg_score': 0,
        'pos_score': 0,
        'key_phrases': [],
        'matched_words': [],
        'sentiment_counts': {'positive': 0, 'negative': 0},
        'analysis': 'No lexicon matches'
    }

    if not text.strip():
        return result

    # Track matched terms and phrases
    matched_terms = []
    key_phrases = []
    
    for sentence in blob.sentences:
        sentence_polarity = 0
        matched_words = []
        
        for word in sentence.words:
            if word in CLIMATE_LEXICON:
                entry = CLIMATE_LEXICON[word]
                sentence_polarity += entry['polarity']
                matched_words.append(word)
                
                matched_terms.append({
                    'word': word,
                    'polarity': entry['polarity'],
                    'category': entry['category'],
                    'neg': entry['neg'],
                    'pos': entry['pos']
                })
        
        if matched_words:
            # Normalize by number of matched words
            sentence_polarity /= len(matched_words)
            
            # Classify phrase
            phrase_category = 'negative' if sentence_polarity < 0 else 'positive'
            result['sentiment_counts'][phrase_category] += 1
            
            key_phrases.append({
                'text': sentence.raw,
                'polarity': sentence_polarity,
                'category': phrase_category,
                'terms': matched_words
            })

    # Calculate final scores
    if matched_terms:
        total_matches = len(matched_terms)
        result.update({
            'score': sum(t['polarity'] for t in matched_terms) / total_matches,
            'neg_score': sum(t['neg'] for t in matched_terms) / total_matches,
            'pos_score': sum(t['pos'] for t in matched_terms) / total_matches,
            'label': 'negative' if result['neg_score'] > result['pos_score'] else 'positive',
            'key_phrases': sorted(key_phrases, key=lambda x: abs(x['polarity'])),
            'matched_words': matched_terms,
            'analysis': (
                f"Matched {total_matches} terms | "
                f"Negative: {result['neg_score']:.2f} | "
                f"Positive: {result['pos_score']:.2f}"
            )
        })

    return result
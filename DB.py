import sqlite3
import Initialization
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def load_data(file_path, sql_where):
    #Load Data
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    cursor.execute("""SELECT COALESCE(Response || ': ' || Comment, Response, Comment) AS Full_Text
                   FROM Response """ + sql_where)

    rows = cursor.fetchall()
    text = []
    for row in rows:
        row = str(row)
        cleaned = row[2:-3]
        text.append(cleaned)
    print(text[0:5])
    cursor.close()
    conn.close()
    return(text)

def export_data(labels, labelset_id, best_probabilities, file_path, sql_where):
    labels = [int(label) for label in labels]

    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    sql_query = "SELECT ResponseID FROM Response" + sql_where
    cursor.execute(sql_query)

    response_ids = cursor.fetchall()


    for i, response_id in enumerate(response_ids):
        cursor.execute("UPDATE Label SET LabelID = ?, Score = ? WHERE LabelsetID = ? AND ResponseID = ?", (labels[i], best_probabilities[i], labelset_id, response_id[0]))

    conn.commit()
    cursor.close()
    conn.close()


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
def pre_process(text):
    cleaned_sentences = []

    replacements = {
        'Naramata': ' ',
        'Penticton': ' '
    }
    text = [sentence.replace(old, new) for sentence in text
                        for old, new in replacements.items()]

    for sentence in text:
        text = re.sub(r'[^A-Za-z\s]', '', sentence)
        cleaned_sentences.append(text)

    preprocessed_texts = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = nltk.WordNetLemmatizer()

    for cleaned_sentence in cleaned_sentences:
        words = nltk.word_tokenize(cleaned_sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        preprocessed_text = ' '.join(lemmatized_words)
        preprocessed_texts.append(preprocessed_text)

    return preprocessed_texts


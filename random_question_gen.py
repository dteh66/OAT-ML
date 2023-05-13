from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
pd.set_option("max_rows", 600)
from pathlib import Path  
import glob

import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag


def generateQuestion(question, fileNames = None):

    assert(type(question) == str)
    #clean
    question = question.replace(".", "").replace("/", "").replace("\"", "").lower()

    #naive numerical
    naive_numerical_question = question.split(" ")

    #TFIDF
    if (fileNames):
        # https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html
        text_files = fileNames
        text_titles = [Path(text).stem for text in text_files]

        tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english')
        tfidf_vector = tfidf_vectorizer.fit_transform(text_files)
        tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names())
        tfidf_df.loc['00_Document Frequency'] = (tfidf_df > 0).sum()
        tfidf_df = tfidf_df.drop('00_Document Frequency', errors='ignore')
        tfidf_df.stack().reset_index()
        tfidf_df = tfidf_df.stack().reset_index()
        tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'term', 'level_2': 'term'})
        tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(60)
        top_tfidf = tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(60)
        # print(top_tfidf)

        # http://librarycarpentry.org/lc-tdm/13-part-of-speech-tagging-text/index.html
        terms = top_tfidf["term"].tolist()
        print(terms)
        print(any(x in question for x in terms))
    # return naive_numerical_question




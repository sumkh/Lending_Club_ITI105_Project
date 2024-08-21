import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import re
from sklearn.preprocessing import OrdinalEncoder


# Custom Transformer for Vector Preprocessing
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=50):
        self.vector_size = vector_size
        self.model = None

    def fit(self, X, y=None):
        # X is the DataFrame containing the text columns
        corpus_list = X.apply(lambda x: x.split()).tolist() # Convert each row to a list of words
        self.model = Word2Vec(
            sentences=corpus_list, # Input corpus: list of sentences
            vector_size=self.vector_size, # Dimensionality of the word vectors
            window=5, # Context window size
            min_count=1, # Minimum frequency count of words to consider
            workers=4 # Number of worker threads for training
        )
        return self

    def get_mean_vector(self, words):
            valid_words = [word for word in words if word in self.model.wv] # Filter out words that are not in the model's vocabulary
            if not valid_words:
                return np.zeros(self.vector_size)
            return np.mean([self.model.wv[word] for word in valid_words], axis=0) # Return vectors from words

    def transform(self, X):
        return pd.DataFrame(X.apply(lambda words: self.get_mean_vector(words.split())).tolist())

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in range(self.vector_size)]


# Define function to convert vector preprocessed data into dataframe
def vec_to_df(X_train, X_train_transformed, vec_preprocessor):
    feature_names = vec_preprocessor.get_feature_names_out()
    pattern = re.compile(r'remainder__')
    feature_names = [pattern.sub('', i ) for i in feature_names]

    # Transform back to dataframe with the features names
    X_train_df = pd.DataFrame(X_train_transformed, columns = feature_names, index=X_train.index)

    return X_train_df

def clean_data(input_data):
    
    # Load input data into a DataFrame
    data = pd.DataFrame(input_data, index=[0])

    # Extract the digits months from 'term'
    data['n_term'] = data['term'].str.extract(r'(\d+)').astype(int)

    # New Feature: Total Instalment Payment
    data['total_installment'] = data['installment'] * data['n_term']

    # New Feature: Total Interest
    data['total_interest'] = data['total_installment'] - data['loan_amnt']

    # New Feature: Number of monthly installment to repay the loan amount assuming no interest component.
    data['n_installmt'] = data['loan_amnt'] / data['installment']

    # Ordinal encode 'grade' according to set order
    grade_unique = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    ordinal_encoder = OrdinalEncoder(categories = [grade_unique])
    data['grade_enc'] = ordinal_encoder.fit_transform(data[['grade']])

    # For 'sub_grade' extract only the digit
    data['sub_grade_enc'] = data['sub_grade'].astype(int)

    # Convert 'emp_title' to lower case
    data['emp_title_clean'] = data['emp_title'].fillna('others')
    data['emp_title_clean'] = data['emp_title_clean'].str.lower()
    data['emp_title_clean'] = data['emp_title_clean'].str.replace(r'[^\w\s]', '', regex=True)

    # Function to remove duplicated words
    def remove_deduplicate(row):
        row = str(row)
        words = row.split()
        unique_words = []
        seen = set()
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        return ' '.join(unique_words)

    # Concat 'title' and 'purpose' and remove duplicate words
    data['title_purpose'] = data['title'] + ' ' + data['purpose']
    data['title_purpose'] = data['title_purpose'].str.lower()
    data['title_purpose'] = data['title_purpose'].str.replace(r'[^\w\s]', '', regex=True) # only characters and space
    data['title_purpose'] = data['title_purpose'].str.replace(r'\b\w{1,2}\b', '', regex=True) # remove words with 1 or 2 characters from a string
    data['title_purpose'] = data['title_purpose'].str.replace(r'\s+', ' ', regex=True) # remove multiple spaces
    data['title_purpose'] = data['title_purpose'].str.strip() # remove leading and trailing spaces
    data['title_purpose'] = data['title_purpose'].str.replace(r'\_', ' ', regex=True) # remove underscores
    data['title_purpose'] = data['title_purpose'].fillna('others')
    data['title_purpose'] = data['title_purpose'].apply(remove_deduplicate)

    # Replace "<1 years" to "0 years" in 'emp_length
    data['emp_length'] = data['emp_length'].str.replace('<1 years', '0 years')

    # Convert 'emp_length' to integer, fillna with '0'
    data['emp_length'] = data['emp_length'].str.extract(r'(\d+)')
    data['emp_length'] = data['emp_length'].fillna(0)
    data['emp_length'] = data['emp_length'].astype(int)

    # For 'home_ownership', group class "ANY", "NONE" into "OTHER"
    data['home_ownership'] = data['home_ownership'].replace(['ANY', 'NONE'], 'OTHER')

    # For 'verification_status' convert to boolean where "Verified" and "Source Verified" is 1
    # and "Not Verified" is 0
    data['verification_status'] = data['verification_status'].map({'Verified':1, 'Source Verified':1, 'Not Verified':0})

    # Convert 'issue_d' into upper case and then to datetime type
    data['issue_d'] = data['issue_d'].str.upper()
    data['issue_d'] = pd.to_datetime(data['issue_d'], format='%b-%Y')

    # Convert 'earliest_cr_line' into upper case and then to datetime type
    data['earliest_cr_line'] = data['earliest_cr_line'].str.upper()
    data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'], format='%b-%Y')

    # Based on 'issued_d' calculate the 'credit_age' by number of years
    data['credit_age'] = (data['issue_d'] - data['earliest_cr_line']).dt.days
    data['credit_age'] = data['credit_age'] / 365

    # New Feature: Total Debt based on 'dti' multiply by 'annual_inc'
    data['total_debt'] = data['dti'] * data['annual_inc']

    # Fill missing values for 'pub_rec', 'mort_acc' and 'pub_rec_bankruptcies'
    data['pub_rec'] = data['pub_rec'].fillna(0)
    data['mort_acc'] = data['mort_acc'].fillna(0)
    data['pub_rec_bankruptcies'] = data['pub_rec_bankruptcies'].fillna(0)

    # Clean 'address' and extract 'state'
    data['state'] = data['address'].str.extract(r'([A-Z]+)\s[0-9]+$')

    # Clean the 'address'
    data['address_clean'] = data['address'].str.lower()
    data['address_clean'] = data['address_clean'].str.replace(r'[^\w\s]', '', regex=True) # only 5-digit number (postal code), characters and space
    data['address_clean'] = data['address_clean'].str.replace(r'\b\w{1,2}\b', '', regex=True) # remove words with 1 or 2 characters
    data['address_clean'] = data['address_clean'].str.replace(r'\s+', ' ', regex=True) # remove multiple spaces
    data['address_clean'] = data['address_clean'].str.strip() # remove leading and trailing spaces
    data['address_clean'] = data['address_clean'].str.replace(r'\_', ' ', regex=True) # remove underscores
    data['address_clean'] = data['address_clean'].apply(remove_deduplicate)

    # Find a 5-digit number (postal code) anywhere in the address from end and extract into 'postal_code'
    data['postal_code'] = data['address_clean'].str.extract(r'(\d{5})[^0-9]*$', expand=False)
    data['postal_code'] = data['postal_code'].str.strip()
    data['postal_code'] = data['postal_code'].fillna('unknown')

    # Extract only text from 'address_clean'
    data['address_text'] = data['address_clean'].str.replace(r'\d+', '', regex=True)
    data['address_text'] = data['address_text'].str.replace(r'[^\w\s]', '', regex=True)
    data['address_text'] = data['address_text'].str.replace(r'\s+', ' ', regex=True) # remove multiple spaces

    # Clean address_text
    data['address_text'] = data['address_text'].str.strip() # remove leading and trailing spaces
    data['address_text'] = data['address_text'].fillna('')

    # Drop features that are no longer required
    data = data.drop(['installment', 'total_installment','loan_amnt',
                        'issue_d', 'earliest_cr_line', 'pub_rec_bankruptcies',
                        'term', 'emp_title', 'purpose', 'title', 'grade',
                        'address', 'address_clean'], axis=1)

    # fillna with 0 for all columns
    data = data.fillna(0)

    return data
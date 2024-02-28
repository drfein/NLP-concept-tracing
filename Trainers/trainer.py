import pandas as pd
import logging
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Enable logging for gensim - optional
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_word2vec_models(years=[]):
    df = pd.read_parquet('ACL-data/acl-publication-info.74k.v2.parquet')

    # Drop rows with None values in 'full_text' column
    df = df.dropna(subset=['full_text'])

    # Preprocess the text
    df['processed_text'] = df['full_text'].apply(simple_preprocess)

    # Convert 'year' column to int
    df['year'] = df['year'].astype(int)

    if not years:
        # Train a Word2Vec model for all years if no specific years are provided
        model = Word2Vec(df['processed_text'].tolist(), min_count=1, workers=4)
        model.save("Models/word2vec.model")
    else:
        # Sort the years in ascending order
        years.sort()

        # Add the minimum and maximum year from the dataframe to the list of years
        years = [df['year'].min()] + years + [df['year'].max()]

        for i in range(len(years) - 1):
            # Filter the dataframe for the papers within the specific interval
            df_interval = df[(df['year'] >= years[i]) & (df['year'] < years[i+1])]

            # Train a Word2Vec model for the papers within the specific interval
            model = Word2Vec(df_interval['processed_text'].tolist(), min_count=1, workers=4)
            model.save(f"Models/word2vec_{years[i]}_to_{years[i+1]}.model")

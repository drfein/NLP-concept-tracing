import pandas as pd
import logging
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import matplotlib as plt

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


def train_word2vec_models_median_split():
    df = pd.read_parquet('ACL-data/acl-publication-info.74k.v2.parquet')

    # Drop rows with None values in 'full_text' column
    df = df.dropna(subset=['full_text'])

    # Preprocess the text
    df['processed_text'] = df['full_text'].apply(simple_preprocess)

    # Convert 'year' column to int
    df['year'] = df['year'].astype(int)

    # Calculate the median number of citations for each year
    median_citations_per_year = df.groupby('year')['numcitedby'].transform('median')

    # Split the dataframe into two based on the median number of citations for each year
    df_less_than_median = df[df['numcitedby'] < median_citations_per_year]
    df_more_than_median = df[df['numcitedby'] > median_citations_per_year]

    # Train a Word2Vec model for the papers with less than the median number of citations
    model_less_than_median = Word2Vec(df_less_than_median['processed_text'].tolist(), min_count=1, workers=4)
    model_less_than_median.save("Models/word2vec_less_than_median.model")

    # Train a Word2Vec model for the papers with more than the median number of citations
    model_more_than_median = Word2Vec(df_more_than_median['processed_text'].tolist(), min_count=1, workers=4)
    model_more_than_median.save("Models/word2vec_more_than_median.model")

def plot_citations():
    df = pd.read_parquet('ACL-data/acl-publication-info.74k.v2.parquet')
    # Only show papers with 0-300 citations
    df = df[df['numcitedby'].between(0, 200)]
    df['numcitedby'].plot(kind='hist', bins=200)  # Increase the number of bins to narrow the intervals
    plt.pyplot.xlabel('Number of Citations')
    plt.pyplot.ylabel('Frequency')
    plt.pyplot.title('Distribution of Citations')
    plt.pyplot.show()

def train_word2vec_models_random_split():
    df = pd.read_parquet('ACL-data/acl-publication-info.74k.v2.parquet')

    # Drop rows with None values in 'full_text' column
    df = df.dropna(subset=['full_text'])

    # Preprocess the text
    df['processed_text'] = df['full_text'].apply(simple_preprocess)

    # Convert 'year' column to int
    df['year'] = df['year'].astype(int)

    # Randomly split the dataframe into two groups that are each 25% of the data
    df_random_split_1 = df.sample(frac=0.5, random_state=1)
    df = df.drop(df_random_split_1.index)
    df_random_split_2 = df

    # Train a Word2Vec model for the first random split
    # model_random_split_1 = Word2Vec(df_random_split_1['processed_text'].tolist(), min_count=1, workers=4)
    # model_random_split_1.save("Models/word2vec_random_split_50_1.model")

    # Train a Word2Vec model for the second random split
    model_random_split_2 = Word2Vec(df_random_split_2['processed_text'].tolist(), min_count=1, workers=4)
    model_random_split_2.save("Models/word2vec_random_split_50_2.model")

def train_word2vec_models_median_year():
    df = pd.read_parquet('ACL-data/acl-publication-info.74k.v2.parquet')

    # Drop rows with None values in 'full_text' column
    df = df.dropna(subset=['full_text'])

    # Preprocess the text
    df['processed_text'] = df['full_text'].apply(simple_preprocess)

    # Convert 'year' column to int
    df['year'] = df['year'].astype(int)

    # Sort the dataframe by year
    df = df.sort_values('year')

    # Find the index of the median paper
    median_index = df.shape[0] // 2

    median_year = df.iloc[median_index]['year']
    print(f"Median year: {median_year}")

    # Split the dataframe into two equal halves
    df_before_median_year = df.iloc[:median_index]
    df_after_median_year = df.iloc[median_index:]

    # Train a Word2Vec model for the papers before the median year
    model_before_median_year = Word2Vec(df_before_median_year['processed_text'].tolist(), min_count=1, workers=4)
    model_before_median_year.save("Models/word2vec_before_median_year.model")

    # Train a Word2Vec model for the papers after the median year
    model_after_median_year = Word2Vec(df_after_median_year['processed_text'].tolist(), min_count=1, workers=4)
    model_after_median_year.save("Models/word2vec_after_median_year.model")
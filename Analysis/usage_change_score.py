import logging
from gensim.models import Word2Vec
from tqdm import tqdm

def detect_usage_change(model_name1, model_name2, topn=1000):
    logging.info('Starting usage change detection...')
    # Load the specified models
    model1 = Word2Vec.load(f"Models/{model_name1}")
    model2 = Word2Vec.load(f"Models/{model_name2}")
    logging.info('Models loaded successfully.')

    # Get the shared vocabulary
    shared_vocab = set(model1.wv.key_to_index).intersection(set(model2.wv.key_to_index))
    logging.info('Shared vocabulary obtained.')

    # Filter out words with less than 100 occurrences in each model
    shared_vocab = {word for word in shared_vocab if model1.wv.get_vecattr(word, 'count') >= 100 and model2.wv.get_vecattr(word, 'count') >= 100}
    logging.info('Filtered shared vocabulary obtained.')

    # Initialize a dictionary to store the usage change scores
    usage_change_scores = {}

    # For each word in the shared vocabulary
    for word in tqdm(shared_vocab, desc="Processing words"):
        # Get the top k nearest neighbors in each model
        nn1 = set([w for w, _ in model1.wv.most_similar(word, topn=topn)])
        nn2 = set([w for w, _ in model2.wv.most_similar(word, topn=topn)])

        # Compute the usage change score
        usage_change_scores[word] = -len(nn1.intersection(nn2))

    # Sort the words by their usage change scores in descending order
    sorted_words = sorted(usage_change_scores.items(), key=lambda x: x[1], reverse=True)

    # Print the top 10 most changed words and their 10 nearest neighbors according to each model
    for word, score in sorted_words[:10]:
        print(f"Word: {word}, Score: {score}")
        print(f"Top 10 nearest neighbors in {model_name1}: {model1.wv.most_similar(word, topn=10)}")
        print(f"Top 10 nearest neighbors in {model_name2}: {model2.wv.most_similar(word, topn=10)}")
        print("\n")

def print_nns(word, model_name1, model_name2, topn=20):
    # Load the specified models
    model1 = Word2Vec.load(f"Models/{model_name1}")
    model2 = Word2Vec.load(f"Models/{model_name2}")

    # Get the shared vocabulary
    shared_vocab = set(model1.wv.key_to_index).intersection(set(model2.wv.key_to_index))

    if word in shared_vocab:
        print(f"Top 10 nearest neighbors in {model_name1}: {model1.wv.most_similar(word, topn=topn)}")
        print(f"Top 10 nearest neighbors in {model_name2}: {model2.wv.most_similar(word, topn=topn)}")
    else:
        print(f"{word} is not in the shared vocabulary.")


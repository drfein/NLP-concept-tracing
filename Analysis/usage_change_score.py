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


def detect_usage_change_multi(models, topn=1000):
    # Load the specified models
    loaded_models = [Word2Vec.load(f"Models/{model_name}") for model_name in models]
    logging.info('Models loaded successfully.')

    # Get the shared vocabulary
    shared_vocab = set.intersection(*[set(model.wv.key_to_index) for model in loaded_models])
    logging.info('Shared vocabulary obtained.')

    # Filter out words with less than 100 occurrences in each model
    shared_vocab = {word for word in shared_vocab if all(model.wv.get_vecattr(word, 'count') >= 100 for model in loaded_models)}
    logging.info('Filtered shared vocabulary obtained.')

    # Initialize a dictionary to store the usage change scores
    usage_change_scores = {}

    # For each word in the shared vocabulary
    for word in tqdm(shared_vocab, desc="Processing words"):
        # Get the top k nearest neighbors in each model
        nns = [set([w for w, _ in model.wv.most_similar(word, topn=topn)]) for model in loaded_models]

        # Compute the usage change score
        max_usage_change = 0
        for i in range(len(nns) - 1):
            usage_change = len(nns[i].intersection(nns[i+1]))
            if usage_change > max_usage_change:
                max_usage_change = usage_change
        usage_change_scores[word] = -max_usage_change

    # Sort the words by their usage change scores in descending order
    sorted_words = sorted(usage_change_scores.items(), key=lambda x: x[1], reverse=True)

    # Print the top 10 most changed words and their 10 nearest neighbors according to each model
    for word, score in sorted_words[:20]:
        print(f"Word: {word}, Score: {score}")
        for i, model in enumerate(loaded_models):
            print(f"Top 10 nearest neighbors in {models[i]}: {[w for w, _ in model.wv.most_similar(word, topn=10)]}")
        print("\n")

    # Loop to print the similarity and neighbors for a word inputted to the terminal
    while True:
        input_word = input("Enter a word to get its similarity and neighbors (or 'exit' to stop): ")
        if input_word.lower() == 'exit':
            break
        if input_word in shared_vocab:
            for i, model in enumerate(loaded_models):
                # The similarity method requires two words to compare, here we are comparing the input_word with itself in each model
                print(f"Similarity of {input_word} in {models[i]}: {model.wv.similarity(input_word, input_word)}")
                print(f"Top 10 nearest neighbors in {models[i]}: {[w for w, _ in model.wv.most_similar(input_word, topn=10)]}")
            print("\n")
        else:
            print(f"{input_word} is not in the shared vocabulary.\n")

def print_nns_multi(word, models, topn=20):
    # Load the specified models
    loaded_models = [Word2Vec.load(f"Models/{model_name}") for model_name in models]

    # Get the shared vocabulary
    shared_vocab = set.intersection(*[set(model.wv.key_to_index) for model in loaded_models])

    if word in shared_vocab:
        for i, model in enumerate(loaded_models):
            print(f"Top 10 nearest neighbors in {models[i]}: {[w for w, _ in model.wv.most_similar(word, topn=topn)]}")
    else:
        print(f"{word} is not in the shared vocabulary.")



import logging
from gensim.models import Word2Vec
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    while True:
        word = input("Enter a word to check its usage change (or 'exit' to stop): ")
        if word == 'exit':
            break
        if word not in shared_vocab:
            print(f"{word} is not in the shared vocabulary.")
            continue

        # Get the top k nearest neighbors in each model
        nn1 = set([w for w, _ in model1.wv.most_similar(word, topn=topn)])
        nn2 = set([w for w, _ in model2.wv.most_similar(word, topn=topn)])

        # Compute the usage change score
        usage_change_score = -len(nn1.intersection(nn2))

        # Print the usage change score and the 10 nearest neighbors according to each model
        print(f"Word: {word}, Usage Change Score: {usage_change_score}")
        print(f"Top 10 nearest neighbors in {model_name1}: {[w for w, _ in model1.wv.most_similar(word, topn=10)]}")
        print(f"Top 10 nearest neighbors in {model_name2}: {[w for w, _ in model2.wv.most_similar(word, topn=10)]}")
        print("\n")

def find_highest_change_words(model_name1, model_name2, topn=1000, num_words=10):
    # Load the specified models
    model1 = Word2Vec.load(f"Models/{model_name1}")
    model2 = Word2Vec.load(f"Models/{model_name2}")

    # Get the shared vocabulary
    shared_vocab = set(model1.wv.key_to_index).intersection(set(model2.wv.key_to_index))

    # Filter out words with less than 100 occurrences in each model
    shared_vocab = {word for word in shared_vocab if model1.wv.get_vecattr(word, 'count') >= 100 and model2.wv.get_vecattr(word, 'count') >= 100}

    # Initialize a dictionary to store the usage change scores
    usage_change_scores = {}

    # For each word in the shared vocabulary
    for word in tqdm(shared_vocab, desc="Processing words"):
        # Get the top k nearest neighbors in each model
        nn1 = set([w for w, _ in model1.wv.most_similar(word, topn=topn) if model1.wv.get_vecattr(w, 'count') >= 100])
        nn2 = set([w for w, _ in model2.wv.most_similar(word, topn=topn) if model2.wv.get_vecattr(w, 'count') >= 100])

        # Compute the usage change score
        usage_change_score = -len(nn1.intersection(nn2))

        # Add the usage change score to the dictionary
        usage_change_scores[word] = usage_change_score

    # Sort the dictionary by the usage change scores in descending order
    sorted_usage_change_scores = sorted(usage_change_scores.items(), key=lambda x: x[1], reverse=True)

    # Write the words with the highest usage change scores and their nearest neighbors to a new file
    with open('highest_change_words.txt', 'w') as f:
        for word, score in sorted_usage_change_scores:
            f.write(f"Word: {word}, Usage Change Score: {score}\n")
            f.write(f"Top 10 nearest neighbors in {model_name1}: {[w for w, _ in model1.wv.most_similar(word, topn=10) if model1.wv.get_vecattr(w, 'count') >= 100]}\n")
            f.write(f"Top 10 nearest neighbors in {model_name2}: {[w for w, _ in model2.wv.most_similar(word, topn=10) if model2.wv.get_vecattr(w, 'count') >= 100]}\n")
            f.write("\n")


def plot_word_change_distribution(model_name1, model_name2, topn=1000):
    # Load the specified models
    model1 = Word2Vec.load(f"Models/{model_name1}")
    model2 = Word2Vec.load(f"Models/{model_name2}")

    # Get the shared vocabulary
    shared_vocab = set(model1.wv.key_to_index).intersection(set(model2.wv.key_to_index))

    # Filter out words with less than 100 occurrences in each model
    shared_vocab = {word for word in shared_vocab if model1.wv.get_vecattr(word, 'count') >= 100 and model2.wv.get_vecattr(word, 'count') >= 100}

    # Initialize a list to store the usage change scores
    usage_change_scores = []

    # For each word in the shared vocabulary
    for word in tqdm(shared_vocab, desc="Processing words"):
        # Get the top k nearest neighbors in each model
        nn1 = set([w for w, _ in model1.wv.most_similar(word, topn=topn)])
        nn2 = set([w for w, _ in model2.wv.most_similar(word, topn=topn)])

        # Compute the usage change score
        usage_change_score = -len(nn1.intersection(nn2))

        # Add the usage change score to the list
        usage_change_scores.append(usage_change_score)

    # Plot the distribution of usage change scores
    plt.hist(usage_change_scores, bins=50)
    plt.xlabel('Usage Change Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Usage Change Scores')
    plt.show()

def find_highest_change_words_multi(models):
    # Initialize a dictionary to store the maximum usage change scores for each word
    max_usage_change_scores = {}

    # For each pair of consecutive models
    for i in range(len(models) - 1):
        model_name1 = models[i]
        model_name2 = models[i+1]

        # Load the specified models
        model1 = Word2Vec.load(f"Models/{model_name1}")
        model2 = Word2Vec.load(f"Models/{model_name2}")

        # Get the shared vocabulary
        shared_vocab = set(model1.wv.key_to_index).intersection(set(model2.wv.key_to_index))

        # Filter out words with less than 100 occurrences in each model
        shared_vocab = {word for word in shared_vocab if model1.wv.get_vecattr(word, 'count') >= 100 and model2.wv.get_vecattr(word, 'count') >= 100}

        # For each word in the shared vocabulary
        for word in tqdm(shared_vocab, desc="Processing words"):
            # Get the top 10 nearest neighbors in each model
            nn1 = set([w for w, _ in model1.wv.most_similar(word, topn=1000) if w in model1.wv.key_to_index])
            nn2 = set([w for w, _ in model2.wv.most_similar(word, topn=1000) if w in model2.wv.key_to_index])

            # Compute the usage change score
            usage_change_score = -len(nn1.intersection(nn2))

            # If the word is not in the dictionary or the usage change score is higher than the current maximum, update the dictionary
            if word not in max_usage_change_scores or usage_change_score > max_usage_change_scores[word][0]:
                max_usage_change_scores[word] = (usage_change_score, model_name1, model_name2)

    # Sort the dictionary by the usage change scores in descending order
    sorted_max_usage_change_scores = sorted(max_usage_change_scores.items(), key=lambda x: x[1][0], reverse=True)

    # Write the words with the highest usage change scores and their nearest neighbors to a new file
    with open('highest_change_words_multi.txt', 'w') as f:
        for word, (score, model_name1, model_name2) in sorted_max_usage_change_scores:
            f.write(f"Word: {word}, Maximum Usage Change Score: {score}, Models: {model_name1}, {model_name2}\n")
            if word in model1.wv.key_to_index:
                f.write(f"Top 10 nearest neighbors in {model_name1}: {[w for w, _ in model1.wv.most_similar(word, topn=10) if w in model1.wv.key_to_index]}\n")
            if word in model2.wv.key_to_index:
                f.write(f"Top 10 nearest neighbors in {model_name2}: {[w for w, _ in model2.wv.most_similar(word, topn=10) if w in model2.wv.key_to_index]}\n")
            f.write("\n")


# plot_word_change_distribution("word2vec_random_split_2.model", "word2vec_random_split_1.model", topn=1000)

def plot_word_change_distribution_multiple(models_pairs, topn=1000):
    """
    Plot the distribution of usage change scores for multiple pairs of models.
    
    Parameters:
    - models_pairs: A list of tuples, where each tuple contains the filenames of two models to be compared.
    - topn: The number of top nearest neighbors to consider for calculating the usage change score.
    """
    # Initialize a figure for plotting
    plt.figure(figsize=(10, 6))

    # Iterate through each pair of models
    for model_names in models_pairs:
        model1 = Word2Vec.load(f"Models/{model_names[0]}")
        model2 = Word2Vec.load(f"Models/{model_names[1]}")
        
        shared_vocab = set(model1.wv.key_to_index).intersection(set(model2.wv.key_to_index))
        shared_vocab = {word for word in shared_vocab if model1.wv.get_vecattr(word, 'count') >= 100 and model2.wv.get_vecattr(word, 'count') >= 100}
        
        usage_change_scores = []
        for word in tqdm(shared_vocab, desc=f"Processing words for {model_names[0]} vs {model_names[1]}"):
            nn1 = set([w for w, _ in model1.wv.most_similar(word, topn=topn)])
            nn2 = set([w for w, _ in model2.wv.most_similar(word, topn=topn)])
            usage_change_score = -len(nn1.intersection(nn2))
            usage_change_scores.append(usage_change_score)
        
        # Plot the distribution for this pair of models
        plt.hist(usage_change_scores, bins=50, alpha=0.5, label=f"{model_names[0]} vs {model_names[1]}")

    # Customize the plot
    plt.xlabel('Usage Change Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Usage Change Scores Across Model Pairs')
    plt.legend(loc='upper right')
    plt.show()

# Example usage:
models_pairs = [
    ("word2vec_random_split_50_1.model", "word2vec_random_split_50_2.model"),
    ("word2vec_more_than_median.model", "word2vec_less_than_median.model"),
]

plot_word_change_distribution_multiple(models_pairs, topn=1000)

from usage_change_score import find_highest_change_words_multi, plot_word_change_distribution

# Define the model names
models = ["word2vec_1952_to_1988.model", "word2vec_1988_to_1994.model", "word2vec_1994_to_2001.model", "word2vec_2001_to_2008.model", "word2vec_2008_to_2022.model"]

# Call the function to detect usage change
# find_highest_change_words_multi(models)

# Plot the distribution of word change for each adjacent era
for i in range(len(models) - 1):
    plot_word_change_distribution(models[i], models[i+1])


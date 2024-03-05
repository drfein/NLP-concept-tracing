from usage_change_score import detect_usage_change_multi, print_nns_multi

# Define the model names
models = ["word2vec_1952_to_1988.model", "word2vec_1988_to_1994.model", "word2vec_1994_to_2001.model", "word2vec_2001_to_2008.model", "word2vec_2008_to_2022.model"]

# Define the word
word = "model"

# Call the function to print nearest neighbors from multiple cohorts for a given one
# print_nns_multi(word, models)

# Call the function to detect usage change
detect_usage_change_multi(models)

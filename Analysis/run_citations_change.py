from usage_change_score import detect_usage_change, find_highest_change_words, plot_word_change_distribution

# Define the model names
model_name1 = "word2vec_less_than_median.model"
model_name2 = "word2vec_more_than_median.model"

# Define the word
word = "model"

# Call the function to detect usage change
# detect_usage_change(model_name1, model_name2)

# Call the function to find the words with the highest usage change
find_highest_change_words(model_name1, model_name2)



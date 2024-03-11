from usage_change_score import detect_usage_change, find_highest_change_words, plot_word_change_distribution

# Define the model names
model_name1 = "word2vec_after_median_year.model"
model_name2 = "word2vec_before_median_year.model"


# Call the function to find the words with the highest usage change
find_highest_change_words(model_name1, model_name2)


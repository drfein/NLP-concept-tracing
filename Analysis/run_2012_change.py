from usage_change_score import detect_usage_change, print_nns

# Define the model names
model_name1 = "word2vec_1952_to_2012.model"
model_name2 = "word2vec_2012_to_2022.model"


# Define the word
word = "model"

# Call the function to print nearest neighbors from two cohorts for a given one
# print_nns(word, model_name1, model_name2)

# Call the function to detect usage change
detect_usage_change(model_name1, model_name2)

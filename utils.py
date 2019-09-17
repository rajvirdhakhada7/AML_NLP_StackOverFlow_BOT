import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'StarSpace_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################
    starspace_embeddings = {}
    file = open(embeddings_path)
    count = 0

    for line in file:
        count = count + 1
        #if(count % 5000 == 0):
        #    print(count)
        line = line.strip().split('\t')
        arr = np.array(line[1:], dtype = np.float32)
        starspace_embeddings[line[0]] = arr

    embeddings_dimension = starspace_embeddings[list(starspace_embeddings)[0]].shape[0]
     
    return starspace_embeddings, embeddings_dimension

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    ########################

    words_sum = np.zeros(dim)
    n_words = 0
    
    for word in question.split():
      if word in embeddings:
        words_sum += embeddings[word]
        n_words += 1
    
    if n_words != 0:
      result = words_sum/n_words
    else:
      result = words_sum
      
    return result

    
def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

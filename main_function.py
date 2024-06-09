"""
This is the main class used to preprocess our complete natural language
processing dataset. In this assignment we use X.com(twitter) tweet's dataset
which contains disaster and non-disaster X's (tweets). As we all konw that 
the twitter data is very unstructured it contains emoticons, links, usernames(mentions) and #hastags
So to make is got fit for our model and remoing all the unwanted words and punchuation, this class
is created.The class contains methods to clean the data, remove stopwords, tokenize and lemmatize the
the data and then get the word embeddings using the Word2Vec model. The class also contains a method
to average the word embeddings to get a single vector representation of the tweet. The class can be used
for any other NLP task as well by changing the data and the model used for word embeddings. It also save the 
model into a folder word2vec_model/<Date_time when model get trained> so that those word2vec models can be reused 
again and again and user can train these models with their own corpus too.
"""

# Importing the required libraries to preprocess the data and save/load the model into/from specific folder.
from datetime import datetime
import os
import re
import string
from wordcloud import STOPWORDS
from nltk.tokenize import TweetTokenizer
import nltk
from gensim.models import Word2Vec
import numpy as np

# ----------------------------------------------------------------------------------------------------------

# Download the wordnet from nltk library 
nltk.download('wordnet')

# ----------------------------------------------------------------------------------------------------------

# Create a class X_clean to preprocess the data and get the word embeddings
class X_clean():
    
    """
        In this class we use two approches so the we can make our code as clean as possible
        1. For creating word2vec model only
	  - For thsi step we pass two variables in our class (1) train = True which is by default False
	  - (2) training_list in which we pass a dataframe column which contain all the tweets,  which is by default None
	  - For training we are not passing any value for variable text, so it will be None
        2. For getting the embeddings of the tweets after training the model on the given text
          - For this we just pass values in the variable text = "Some text" to preprocessed
	  In this process we use the model which we already trained on our training courpus
    """
#    __init__ method to initialize the class variables

    def __init__(self, text=None, train=False, training_list=None):
        self.text = text
        self.train = train
        self.training_list = training_list
    
#     Method 1. for clean the data, In this method we remove the links, HTML's, replacing '%20 with _', 
#     emoticons, usernames, hashtags and punctuations from the text. 
    def data_clean(self):
        emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
        if self.train == True:
              text_list = []
              for lst in self.training_list:
                  text = re.sub(r'https?://\S+|www\.\S+',"",lst)   # Remove links
                  text = re.sub(r'<.*?>',"",text)   # Remove HTML parts
                  text = re.sub(r"\x89ÛÒ", "", text) # Remove the "\x89ÛÒ" special character from the text
                  text = emoji_pattern.sub(r'', text) # Remove emoticons
                  text = text.replace("%20", "_")     # Replace the %20 with _
                  text = re.sub(r"@[^\s]+[\s]?", "", text)   # Remove usernames
                  table = str.maketrans('', '', string.punctuation)
                  text = text.translate(table) # Remove punctuations
                  text_list.append(text)
              return text_list
        else:
             text = self.text
             text = re.sub(r'https?://\S+|www\.\S+',"",text)  # Remove links
             text = re.sub(r'<.*?>',"",text) # Remove HTML parts
             text = re.sub(r"\x89ÛÒ", "", text) # Remove the "\x89ÛÒ" special character from the text
             text = emoji_pattern.sub(r'', text) # Remove emoticons
             text = text.replace("%20", "_") # Replace the %20 with _
             text = re.sub(r"@[^\s]+[\s]?", "", text)	# Remove usernames
             table = str.maketrans('', '', string.punctuation)
             text = text.translate(table) 		# Remove punctuations
             return text
    
#     This Method remove all the stopwords from the data. We are using the STOPWORDS from the wordcloud library
#     to remove the stopwords from the data. The input for this method is the output of previous method
    def remove_stopwords(self):
        # If it's training data then it will return the list of cleaned text else it will return the cleaned text
        if self.train == True: 
          text_list = self.data_clean()    # Using the output from data_clean() method
          text_list = [' '.join([word for word in text.split() if word not in STOPWORDS]) for text in text_list]
          return text_list
        else:
          text = self.data_clean()  # Use the output of data_clean() method
          text = ' '.join([word for word in text.split() if word not in STOPWORDS])
          return text
    
    # This method tokenize the data and lemmatize the data. We are using the TweetTokenizer from nltk library 
    # to tokenize the data and WordNetLemmatizer from nltk library to lemmatize the data. The input for this method
    # is the output of remove_stopwords() method. 
    def tokenize_and_lemmatize(self):
        tokenizer = TweetTokenizer()   # Using the TweetTokenizer from nltk library to tokenize the data
        lemmatizer = nltk.stem.WordNetLemmatizer()  # Using the WordNetLemmatizer from nltk library to lemmatize the data
        # if teain in True then it will retun a nested list of data contain  both lemmatize and tokenized data
        if self.train == True:
            text_list = self.remove_stopwords()   # Using the output of remove_stopwords() method
            text_list = [text.lower() for text in text_list]    # lower case the each word of the data
            tokens = [tokenizer.tokenize(text) for text in text_list]   # Tokenize the data
            text_list = [[lemmatizer.lemmatize(token) for token in tokens] for tokens in tokens]   # lemmatization 
            return text_list
        else:
            text = self.remove_stopwords()  # Use the output of remove_stopwords() method
            text = text.lower()    # lowercase the text
            tokens = tokenizer.tokenize(text)   # Tokenize the string
            text = [lemmatizer.lemmatize(token) for token in tokens]   # lemmatize the string and store it into a list in a variable
            return text

    # This method is used to train the Word2Vec model on the given text data. If the train variable is True then
    # it will train the Word2Vec model on the given text data and save the model into a folder word2vec_model/<Date_time>
    # If the train variable is False then it will load the latest trained model and get the embeddings of the text data.
    # The input for this method is the output of tokenize_and_lemmatize() method.
    # This is the method which create the embeddings of all the unique words available in our data into a 256 dimension vector
    def get_embedding(self):
        # If it's training data then it will train the Word2Vec model else it will load the latest trained model
        if self.train:
            text_list = self.tokenize_and_lemmatize()  # Use the output of tokenize_and_lemmatize() method
            # Train Word2Vec model if it's training data, vector size 265, window size 5, min_count 3, workers 6
	    # Window means the number of words before and after a target variable, min_count meains at least 
	    # that word must be there atleast 3 time, workers is the number of CPU threads it can use  
            model = Word2Vec(vector_size=256, window=5, min_count=3, workers=6)
            # Build the vocabulary and train the model on the data
            model.build_vocab(text_list)
            model.train(text_list, total_examples=model.corpus_count, epochs=model.epochs)
            
            # Save the trained model
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # To get the current date and time
            model_path = os.path.join("word2vec_model", current_time)  # To create the path to save the model
            # If the model_path is not exit then it will first create it and then save the model there    
            if not os.path.exists(model_path):
                os.makedirs(model_path)
	    # Saving the model
            model.save(os.path.join(model_path, "word2vec_model"))
            return "Training Done"
        else:
            text = self.tokenize_and_lemmatize()  # Use the output of tokenize_and_lemmatize() method
            # Load the latest trained model
            latest_model_path = max([os.path.join("word2vec_model", d,"word2vec_model") for d in os.listdir("word2vec_model")], key=os.path.getctime)
            model = Word2Vec.load(latest_model_path)
            embeddings = []
            # Get embeddings for each token in the text       
            for token in text:
                #  Out of Vocabulary (OOV) Handleing, if the words are OOV then we just skipped that word
                try: 
                     # Try to get embedding for the token directly
                     embedding = model.wv[token]
                     embeddings.append(embedding)
                except KeyError:# If token is OOV, skip it
                     pass
            return embeddings

    # This method is used to average the word embeddings to get a single vector representation of the tweet, So, that we can 
    # Feed them into our model to train it and make predictions. 
    # The input for this method is the output of get_embedding() method.
    def average_embed(self):
        #  This method is not used in training, it's a method to get the embeddings of the text data
        #  which was already processed by get_embedding() method
        embeddings = self.get_embedding()  # Use the output of get_embedding() method
        # To Get a 256 dim vector of the tweet
        vector_size = 256
        # Initialize the embedding vector with zeros
        embedding = np.zeros(vector_size)
        # Add up all the vectors from the embeddings and get the average of all the vectors
        wcount = 1
        for text in embeddings:
            embedding += text
            wcount += 1
        embedding = embedding / wcount   # Get the average of all the vectors by dividing them with the total count in that particular text
        return embedding
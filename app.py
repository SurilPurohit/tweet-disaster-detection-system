"""
This is the main file for the streamlit app. It contains the code for the streamlit app.
https://tweetapp.streamlit.app
"""

# Importing the necessary libraries
import streamlit as st
import pickle

# User defined Class importing for cleaning the text
from main_function import X_clean

# Function to predict the tweet and display the result
def prediction(user_tweet, model):
    tweet_clean = X_clean(text = user_tweet, train = False)   # Create an instance of our class and pass tweet into it 
    user_tweet = tweet_clean.average_embed()   # Get the average embedding of the tweet
    predicted_output = model.predict([user_tweet])  # Predict the tweet using the trained model
    if predicted_output == 1:
        print (st.error('This tweet is about a disaster'))
    else:
        print(st.success("This tweet isn't about a disaster"))

# Setting the page config for the WebApp
st.set_page_config(page_title = "Disaster Detection App", page_icon = "Images/logo.png")

# Title of our WebApp
st.title('Disaster Detection App')

#  Sidebar for the WebApp, where the user can select the model which they want to use for Prediction
menu = st.sidebar.radio("Menu",["Home",
                                "Ada Boost",
                                "Bernoulli Naive Bayes",
                                "Cat Boost",
                                "Decision Tree",
                                "Extra Tree",
                                "Gradient Boosting",
                                "K Nearest Neighbors",
                                "Logistic Regression",
                                "Random Forest",
                                "Ridge Classifier",
                                "XG Boost"])

#  Conditional statements for the WebApp sidebar based on whether a new model or an existing model is selected 
if menu=="Home":
    st.image("images/x.jpg",width=550)
    st.write("Welcome to the Disaster Detection App")
    st.write("This app is designed to predict whether a tweet is about a disaster or not.")
    st.write("Please select a model from the sidebar to get started.")
# ========================================================================================================
elif menu=="Ada Boost":
    # Load the trained AdaBoost model
    with open('trained_models/AdaBoostClassifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Ada Boost Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
# ========================================================================================================
elif menu=="Bernoulli Naive Bayes":
    # Load the trained Bernoulli Naive Bayes model
    with open('trained_models/BernoulliNB.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Bernoulli Naive Bayes Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
# ========================================================================================================
elif menu=="Cat Boost":
    # Load the trained Cat Boost model
    with open('trained_models/CatBoostClassifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Cat Boost Classifier Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
# ========================================================================================================
elif menu=="Decision Tree":
    # Load the Decision Tree model
    with open('trained_models/DecisionTreeClassifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Decision Tree Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
# ========================================================================================================
elif menu=="Extra Tree":
    # Load the Extra Tree model
    with open('trained_models/ExtraTreeClassifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Extra Tree Classifier Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
# ========================================================================================================
elif menu=="Gradient Boosting":
    # Load the Gradient Boosting model
    with open('trained_models/GradientBoostingClassifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Gradient Boosting Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
# ========================================================================================================
elif menu=="K Nearest Neighbors":
    # Load the KNN model
    with open('trained_models/KNeighborsClassifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('K Nearest Neighbors Classifier Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet') 
# ========================================================================================================
elif menu=="Logistic Regression":
    # Load the Logistic Regression model
    with open('trained_models/LogisticRegressionCV.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Logistic Regression Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet') 
# ========================================================================================================
elif menu=="Random Forest":
    # Load the Random Forest Classifier model
    with open('trained_models/RandomForestClassifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Random Forest Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
# ========================================================================================================
elif menu=="Ridge Classifier":
    # Load the Ridge Classifier model
    with open('trained_models/RidgeClassifierCV.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('Ridge Classifier Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
# ========================================================================================================
elif menu=="XG Boost":
    # Load the XGBoost classifier model
    with open('trained_models/XGBClassifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    # Display the model name
    st.header('XGBoost Classifier Model')
    # Text area for the user to enter the tweet
    tweet = st.text_area('Enter Tweet')
    # Button to predict the tweet
    if st.button('Predict'):
        if tweet:
            prediction(tweet, loaded_model)
        else:
            st.error('Please enter a tweet')
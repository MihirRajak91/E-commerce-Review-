import numpy as np 
import pandas as pd 
import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

st.title('E-commerce review classification')
st.write("---")
#st.subheader('Cleaned DataSet of review data of Amazon obatined from ')
st.markdown("DataSet of reviews of Amazon products obatined from [Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/code)")
st.text('the data obtained from Kggle was cleaned and preprocessed')
df = pd.read_csv('cleaned_test.csv')

#-------------------------------------------------------model creation

# Training model
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer()
X = df['processed_text_column']
Y = df['Rating']
df
st.write("---")

st.subheader('Review Classification')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

#Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=2000)),
                     ('clf', LogisticRegression())])
model = pipeline.fit(X, Y)

code = """
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer()
X = df['processed_text_column']
Y = df['Rating']
df

st.subheader('Review Classification')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

#Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=2000)),
                     ('clf', LogisticRegression())])
model = pipeline.fit(X, Y)"""
st.code(code,language="python")

st.write("---")
#-----------------------------------------------------model creation end
with st.form(key='my_form'):
    Review_data = st.text_input('Input Review')
    submit_button = st.form_submit_button('Predict')
try:
    if submit_button:
        r_data = {'predict_news':["Review_data"]}
        review_data_df = pd.DataFrame(r_data)
        review_review_cat = model.predict(review_data_df['predict_news'])
    #st.write("Predicted news category = ",review_review_cat[0])
    st.write("---")
    if review_review_cat == 1:
        st.write("The provided review is")
        st.write("NEGATIVE")
    else:
        st.write("The provided review is:")
        st.write("POSITIVE")

except:
    print("error")




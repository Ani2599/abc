import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()
def transformation_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()        
            
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)  

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('pytnon model.pkl','rb'))

st.title('Email spam classifier')

input_sms = st.text_area('enter the sms')

if st.button('Predict'):
    

 transformed_sms = transformation_text(input_sms)

 vector_input = tfidf.transform([transformed_sms])

 result = model.predict(vector_input)[0]

 if result == 1:
    st.header('Spam')
 else:
    st.header("Not Spam")
    
    
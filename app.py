import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("Fake News Detection")

news = st.text_area("Enter News Article")

if st.button("Check News"):
    
    vector = vectorizer.transform([news])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("This is Real News")
    else:
        st.error("This is Fake News")
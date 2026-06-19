import requests
import streamlit as st

st.title("FastAPI + Streamlit Demo")

user_text = st.text_input("Enter some text")

if st.button("Submit"):

    response = requests.post(
        "http://gemma-fastapi:8000/process",
        json={"text": user_text}
    )

    if response.status_code == 200:
        result = response.json()

        st.success("Response received")
        st.write("Original:", result["original"])
        st.write("Processed:", result["processed"])
    else:
        st.error("API call failed")

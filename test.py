import streamlit as st

st.text_input("Your name", key="name")

# This exists now:
st.session_state.name


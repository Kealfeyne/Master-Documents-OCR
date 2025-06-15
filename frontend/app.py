import os
import requests as rq
import streamlit as st


# Внутрення сеть докер композа
st.write(rq.get(f"http://backend:{os.getenv('BACKEND_PORT')}/").text)

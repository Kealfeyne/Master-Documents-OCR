import os
import requests as rq
import streamlit as st
import base64
import json
from streamlit_autorefresh import st_autorefresh


# @st.fragment
def refresh():
    response = rq.post(f"{URL}/refresh/").content
    dict_response = json.loads(response.decode('utf-8'))

    return dict_response

st.set_page_config(layout="wide")
# refresher = st_autorefresh(interval=5000, limit=None, key="fizzbuzzcounter")

URL = f"http://backend:{os.getenv('BACKEND_PORT')}"
# URL = f"http://localhost:{os.getenv('BACKEND_PORT')}"

uploaded_file = st.file_uploader('Фотография', accept_multiple_files=False)

if uploaded_file:
    source_img = uploaded_file.read()
    response = rq.post(f"{URL}/detect/", files={"file": source_img})
    # print(response)

dict_response = refresh()
print(dict_response)

if dict_response != {}:
    image = base64.b64decode(dict_response["bimage"])
    text = dict_response["text"].replace("\n", "  \n")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image)

    with col2:
        st.markdown(text)


# if refresher % 1 == 0:
#     response = rq.post(f"{URL}/refresh/").content
#     dict_response = json.loads(response.decode('utf-8'))
#     print(dict_response)
#     # print(dict_response["text"])

#     if dict_response != {}:
#         image = base64.b64decode(dict_response["bimage"])
#         text = dict_response["text"].replace("\n", "  \n")
#         print(text)

#         col1, col2 = st.columns(2)

#         with col1:
#             st.image(image)

#         with col2:
#             st.markdown(text)

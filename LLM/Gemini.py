import os
import google.generativeai as genai
import streamlit as st

from time import sleep


def configure_api(api_key, proxy_url=None):
    if proxy_url:
        os.environ['https_proxy'] = proxy_url if proxy_url else None
        os.environ['http_proxy'] = proxy_url if proxy_url else None
    genai.configure(api_key=api_key)
    
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
# os.environ['GOOGLE_API_KEY'] ="AIzaSyBaz13UTSLEsag18c_rHQ9yFUbX4sx3YYM"
    
class Gemini:
    def __init__(self, model_path='gemini-1.5-flash', api_key='AIzaSyCEzc2NtaIa3eBMh5QNp1wDaeSCH0OrN-g', proxy_url=None):
        configure_api(api_key, proxy_url)
        self.model = genai.GenerativeModel(model_path)
        
    def generate(self, question):
        response = self.model.generate_content(question, stream = True)
        for res in response:
            if hasattr(res, 'text') and res.text:
                yield res.text
        # print(f"{'*'*20}")   
 
 
        
def test():
    llm = Gemini()
    
    for answer in llm.generate("who are you"):
        print(answer)
        
    # sleep(5)
    for answer in llm.generate("who designed you?"):
        print(answer)

 
    for answer in llm.generate("what are you capable of?"):
        print(answer)


if __name__ == '__main__':
    test()

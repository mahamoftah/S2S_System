import streamlit as st
import os
from st_audiorec import st_audiorec
from streamlit_TTS import text_to_speech
from LLM.Embedding import *
# from LLM.Gemini import *
# from LLM.GroqApi import *
from STT.GroqApiSTT import *
from groq import Groq
import google.generativeai as genai
import re

# Define model options
modelOptions = {
    'Gemma2 9b': 'gemma2-9b-it',
    "Gemma 7b": 'gemma-7b-it',
    'Gemini': 'gemini-1.5-flash',
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "LLaMA3 70b": "llama3-70b-8192",
    "LLaMA3 8b": "llama3-8b-8192",
}

# st.logo("D:/2. GP/images/image.png")
# Dropdown menu for model selection
selected_model = st.sidebar.selectbox("Select a model", list(modelOptions.keys()))
selected_model_id = modelOptions[selected_model]

if selected_model_id == 'gemini-1.5-flash':
    chatModel = genai.GenerativeModel('gemini-1.5-flash')
else:
    chatModel = Groq()

STTModel = GroqSTT()
Vectoriser = PDFVectoriser()

languages = ['English', 'Arabic']
language = st.sidebar.selectbox("Choose a language", languages)
lang = language[:2].lower()

# HTML and CSS for styled title
title_html = """
    <style>
    .title {
        font-size: 70px;
        font-weight: 800;
        color: #c13584; /* Gradient color start */
        background: -webkit-linear-gradient(#4c68d7, #ff6464); /* Gradient background */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 50px;
        font-weight: 400;
        color: #333337; /* Subtitle color */
    }
    </style>
    <div class="title">Hello,</div>
    <div class="subtitle">How can I help you today?</div>
    """

st.markdown(title_html, unsafe_allow_html=True)

# Initialize session state variables
if "pdf" not in st.session_state:
    st.session_state.pdf = None
if "v_db" not in st.session_state:
    st.session_state.v_db = None
if "texts" not in st.session_state:
    st.session_state.texts = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar to switch between text and audio interaction
interaction_mode = st.sidebar.radio("Interaction Mode", ["Text", "Audio"])

# Shared file upload functionality
st.sidebar.title("Chatbot")
pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if pdf and st.sidebar.button("Create Vector Database"):
    with st.spinner("Creating vector database..."):
        texts = Vectoriser.split_text(Vectoriser.extract_from_pdf(pdf))
        if not texts:
            st.error("No texts were extracted from the PDF.")
        else:
            st.session_state.v_db = Vectoriser.create_vector_db(texts)
            st.session_state.pdf = pdf
            st.session_state.texts = texts
            if st.session_state.v_db is not None:
                st.success("Vector database created successfully!")
            else:
                st.error("Failed to create vector database.")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat history cleared!")

if st.sidebar.button("Delete Vector Database"):
    st.session_state.v_db = None
    st.session_state.pdf = None
    st.session_state.texts = None
    st.sidebar.success("Vector database deleted!")

if interaction_mode == "Text":
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Enter your message:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        placeholder = st.chat_message("AI").empty()
        similar_text = "You are a Multi Task AI Agent"

        if st.session_state.v_db:
            similar_context = Vectoriser.get_similar_context(user_input, 5, st.session_state.v_db)

            for doc in similar_context:
                similar_text += doc.page_content

        with st.spinner("Thinking..."):
            stream_res = ""
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            combined_input = f"{conversation_history}\nuser: {user_input}\nAI:"
            combined_input += similar_text

            if selected_model_id == 'gemini-1.5-flash':
                response = chatModel.generate_content(combined_input, stream=False).text
            else:
                quest = [{
                    "role": "user",
                    "content": combined_input,
                }]

                response = chatModel.chat.completions.create(
                    messages=quest,
                    model=selected_model_id,
                    temperature=0,
                    stream=False,
                )
                response = response.choices[0].message.content

            st.session_state.messages.append({"role": "AI", "content": response})

elif interaction_mode == "Audio":

    st.title("Voice Interaction")

    # Recording audio
    st.subheader("Record Your Message:")
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        with open("recorded_audio.wav", "wb") as f:
            f.write(wav_audio_data)

        # Get the URL for the audio data
        audio_url = st.audio(wav_audio_data, format="audio/wav")

        # Use custom HTML and JavaScript to autoplay the audio and make it invisible
        st.markdown(f"""
        <audio id="audio" autoplay>
            <source src="{audio_url}" type="audio/wav">
        </audio>
        <script>
            var audio = document.getElementById('audio');
            audio.style.display = 'none';
            audio.play();
        </script>
        """, unsafe_allow_html=True)

        transcription = STTModel.transcribe_audio("recorded_audio.wav", lang)
        user_input = transcription

        st.session_state.messages.append({"role": "user", "content": user_input})
        similar_text = "You are a Multi Task AI Agent"

        if st.session_state.v_db:
            similar_context = Vectoriser.get_similar_context(user_input, 5, st.session_state.v_db)
            for doc in similar_context:
                similar_text += doc.page_content

        with st.spinner("Thinking..."):
            stream_res = ""
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            combined_input = f"{conversation_history}\nuser: {user_input}\nAI:"
            combined_input += similar_text

            if selected_model_id == 'gemini-1.5-flash':
                response = chatModel.generate_content(combined_input, stream=False).text
            else:
                quest = [{
                    "role": "user",
                    "content": combined_input,
                }]

                response = chatModel.chat.completions.create(
                    messages=quest,
                    model=selected_model_id,
                    temperature=0,
                    stream=False,
                )
                response = response.choices[0].message.content

            st.session_state.messages.append({"role": "AI", "content": response})

            pattern = re.compile(r'[*#,]')
            text = pattern.sub('', response)
            text_to_speech(text=response, language=lang)

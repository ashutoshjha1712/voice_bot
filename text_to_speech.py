import streamlit as st
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import numpy as np
import requests
import speech_recognition as sr  # For Speech-to-Text
from gtts import gTTS  # For Text-to-Speech
import os
import uuid

# Hugging Face API function for Mistral
def query_mistral(payload):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    API_TOKEN = "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"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            return None

        data = response.json()
        generated_text = data[0]['generated_text']
        prompt_index = generated_text.find('[/INST]')
        if prompt_index != -1:
            generated_text = generated_text[prompt_index + len('[/INST]'):].strip()
        return generated_text
    except requests.exceptions.RequestException:
        return None
    except ValueError:
        return None

def using_mistral(query_text):
    prompt = f"""[INST] You are an assistant named Ashutosh, and you are sitting in an interview. 
    You must answer the questions based **only on the given data** in a formal and polite manner. 
    Keep responses concise and to the point.

    ## Candidate Information:
    **Name:** Ashutosh  
    **Profile:** Software Engineer (Generative AI)  
    **Years of Experience:** 2 years  
    **Company:** Appolo Computers Pvt Ltd  
    **Email:** jhaa9696@gmail.com  
    **LinkedIn:** www.linkedin.com/in/kumar-ashutosh17  
    **GitHub:** https://github.com/ashutoshjha1712  

    ## **Technical Skills:**
    - **Programming:** Python
    - **Data Storage & Search:** Elasticsearch
    - **Machine Learning Frameworks:**  Hugging Face, NLTK
    - **Deployment & Integration:** FastAPI, Docker, RESTful API Development, IBM Cloud
    - **Generative AI:** Mistral, BART, BERT, LangChain, Retrieval-Augmented Generation (RAG)
    - **Deep Learning:** Transformer Models (mBERT, BERT-based)
    - **NLP Techniques:** Named Entity Recognition (NER), Text Summarization, Sentiment Analysis
    - **Vector Databases & Search:** Implementing efficient search capabilities in large-scale enterprise data

    ## **Projects & Experience:**
    1️ **Searching Over Enterprise Data**  
       - Built a **question-answering system** using NLP and Generative AI models.  
       - Utilized **Elasticsearch & vector databases** for efficient enterprise data retrieval.  
       - Technologies: **Mistral Model, Hugging Face, LangChain, Elasticsearch, RAG**  

    2️ **SummarizeIQ**  
       - Developed a **document summarization system** for enterprise use.  
       - Used **BART Model & Elasticsearch** for indexing and summarizing large documents.  
       - Technologies: **BART Model, MinIO, Elasticsearch**  

    3️ **Context-Based Sentiment & Route Prediction**  
       - Created a **sentiment prediction model** using transformer-based Generative AI models.  
       - Processed diverse text inputs with **contextual awareness** for better sentiment analysis.  
       - Technologies: **mBERT, BERT-based models, Kaggle Dataset**  

    ## **Superpowers:**
    - **Flashbacks:** Understanding the importance of reviewing past results to improve future decisions.  
    - **Instant Insight & Clarity:** Quickly identifying key points, simplifying complex problems, and improving decision-making.  
    - **MindMap:** A unique blend of **strategic thinking, adaptability, and problem-solving**.  

    ## **Areas of Growth:**
    1. **Technical:** Building more scalable and efficient solutions.  
    2. **Management:** Leadership and teamwork in multi-disciplinary environments.  
    3. **System Design:** Developing scalable, high-performance AI-driven applications.  

    **Interviewer Question:** {query_text}[/INST]
    """
    
    max_new_tokens = 500
    return query_mistral({"parameters": {"max_new_tokens": max_new_tokens}, "inputs": prompt})


# Function to transcribe audio using SpeechRecognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Could not request results, check your internet connection"

# Function to convert text to speech using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    unique_filename = f"response_{uuid.uuid4().hex}.mp3"
    temp_audio_file = os.path.abspath(unique_filename)  
    temp_audio_file = os.path.normpath(temp_audio_file)
    tts.save(temp_audio_file)
    return temp_audio_file

# Streamlit UI
st.title("🎙️ Audio Q&A with Mistral AI & Speech Output")

# Record Audio
audio_bytes = audio_recorder(
    text="🎤 Click to record your question",
    recording_color="#e63946",
    neutral_color="#457b9d",
    icon_name="microphone",
    icon_size="4x",
    pause_threshold=2.0,
    sample_rate=41000
)

# Save and Process Audio
if audio_bytes:
    audio_path = "recorded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    st.success("✅ Audio Recorded Successfully!")
    st.audio(audio_bytes, format="audio/wav")

    # Transcribe Audio
    with st.spinner("Transcribing audio..."):
        transcribed_text = transcribe_audio(audio_path)
        st.write(f"**Transcribed Text:** {transcribed_text}")

    # Call Mistral LLM for an Answer
    if transcribed_text:
        print(f"transcribe text:{transcribed_text}")
        with st.spinner("Getting AI Response..."):
            ai_response = using_mistral(transcribed_text)
            print(ai_response)
            if ai_response:
                st.success("✅ AI Response Generated")
                st.write(f"**Mistral AI Answer:** {ai_response}")

                # Convert AI response to speech
                speech_file = text_to_speech(ai_response)
                st.audio(speech_file, format="audio/mp3")

                st.success("🎧 AI response converted to speech!")
            else:
                st.error("❌ Failed to get a response from Mistral AI")

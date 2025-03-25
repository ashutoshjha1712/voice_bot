import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import ollama  
import requests
import uuid
import os
import time

# Initialize session state for stop button
if "stop" not in st.session_state:
    st.session_state.stop = False

# Function to recognize speech
def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening... Please speak.")

        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
            if st.session_state.stop:
                return "Stopped by user."
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "Speech recognition service is unavailable."
        except Exception as e:
            return f"Error: {str(e)}"

# Function to query Mistral model
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
    prompt = f"[INST] Answer the following question based on your knowledge base: {query_text}[/INST]"
    max_new_tokens = 500
    return query_mistral({"parameters": {"max_new_tokens": max_new_tokens}, "inputs": prompt})

# Function for text-to-speech (TTS)
def text_to_speech(text):
    if st.session_state.stop:  # Stop if stop button is clicked
        return

    tts = gTTS(text)
    unique_filename = f"response_{uuid.uuid4().hex}.mp3"
    temp_audio_file = os.path.abspath(unique_filename)  
    temp_audio_file = os.path.normpath(temp_audio_file)  # Normalize path for Windows compatibility
    tts.save(temp_audio_file)

    if not st.session_state.stop:
        st.audio(temp_audio_file)  # Use Streamlit's built-in audio player

    # Delete temp file after playback (optional)
    time.sleep(2)  # Wait for playback to start
    os.remove(temp_audio_file)

# Streamlit UI
st.title("🎙️ Local AI Voice Assistant (Ollama)")

# Stop Button (Always Visible)
stop_placeholder = st.empty()
if stop_placeholder.button("🛑 Stop"):
    st.session_state.stop = True  # Set stop state
    st.warning("🔴 Stopped! You can now ask a new question.")
    st.rerun()  # Refresh UI immediately

# Button to start recording
if st.button("🎤 Speak Now"):
    st.session_state.stop = False  # Reset stop state
    user_input = speech_to_text()
    st.success(f"🗣️ You said: {user_input}")

    if user_input and not st.session_state.stop:
        response_placeholder = st.empty()  # Dynamic placeholder for response
        response = using_mistral(user_input)
        response_placeholder.success(f"🤖 Model said: {response}")
        text_to_speech(response)


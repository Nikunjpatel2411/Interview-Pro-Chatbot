import os
import json
import random
import ssl
import nltk
import streamlit as st
from gtts import gTTS
import playsound
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from googletrans import Translator

# Setup for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Initialize Translator
translator = Translator()

# Load intents from JSON
file_path = "intents.json"
try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except FileNotFoundError:
    st.error("Intents file not found. Please upload it to the correct location.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding the intents JSON file. Please check its format.")
    st.stop()

# Preprocess and Train
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, y)


# Format response
def format_response(intent):
    response = random.choice(intent['responses'])
    if 'additional_info' in intent:
        if 'examples' in intent['additional_info']:
            examples = "\n".join(
                [f"Q: {ex['question']} - A: {ex.get('sample_answer', ex.get('suggestion', ''))}" for ex in
                 intent['additional_info']['examples']])
            response += f"\n\nExamples:\n{examples}"
        if 'resources' in intent['additional_info']:
            resources = "\n".join([f"{res['topic']}: {res['url']}" for res in intent['additional_info']['resources']])
            response += f"\n\nResources:\n{resources}"
    return response


# Chatbot Response
def chatbot(input_text, user_lang):
    try:
        input_text_en = translator.translate(input_text, src=user_lang, dest='en').text
        input_text_vector = vectorizer.transform([input_text_en.lower()])
        predicted_tag = clf.predict(input_text_vector)[0]

        for intent in intents:
            if intent['tag'] == predicted_tag:
                response = format_response(intent)
                response_translated = translator.translate(response, src='en', dest=user_lang).text
                return response_translated

    except Exception as e:
        return f"I couldn't process your request. Please try again. ({str(e)})"


# Voice Output
def speak_response(response):
    temp_dir = os.path.join(os.getcwd(), 'temp_audio')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_audio_path = os.path.join(temp_dir, "response.mp3")

    try:
        tts = gTTS(response)
        tts.save(temp_audio_path)
        playsound.playsound(temp_audio_path)
        os.remove(temp_audio_path)
    except Exception as e:
        print(f"Error during speech synthesis or playback: {str(e)}")


# Speech Input using SpeechRecognition
def listen_to_user():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                user_input = recognizer.recognize_google(audio)
                st.success(f"You said: {user_input}")
                return user_input
            except sr.UnknownValueError:
                st.error("Sorry, I did not understand that.")
                return None
            except sr.RequestError as e:
                st.error(f"Error with the speech recognition service: {e}")
                return None
    except AttributeError as e:
        if "PyAudio" in str(e):
            st.warning("This feature is unavailable because PyAudio is not installed. Please check your setup.")
        else:
            st.error(f"An unexpected error occurred: {e}")
        return None


# Main Application
def main():
    st.sidebar.image("chatbot_logo.png", width=150)
    menu_options = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Menu", menu_options)

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # Home Section
    if choice == "Home":
        st.markdown("<h1 style='text-align: center;'>Job Interview Preparation Chatbot</h1>", unsafe_allow_html=True)

        st.sidebar.title("Language Selection")
        language_options = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "Hindi": "hi",
            "German": "de",
            "Chinese (Simplified)": "zh-cn"
        }
        user_lang = st.sidebar.selectbox("Choose your language:", list(language_options.keys()))
        user_lang_code = language_options[user_lang]

        for chat in st.session_state.chat_log:
            if chat['sender'] == 'user':
                st.markdown(
                    f"<div style='padding:10px; background-color:#262730; border-radius:10px;color:#edf0eb;margin-bottom:5px;'><B>You:</B> {chat['message']}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='padding:10px; background-color:#E5E5E5; border-radius:10px;color:#020500; margin-bottom:5px;'><B>Bot:</B> {chat['message']}</div>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns([8, 2])
        with col1:
            user_input = st.text_input("Type your message or click the mic to speak:", key="user_input_home",
                                       label_visibility="collapsed")
        with col2:
            speak_button = st.button("Speak", use_container_width=True, disabled=False)
            if speak_button:
                user_input = listen_to_user()  # Capture voice input when the user clicks the "Speak" button
                if user_input:
                    st.session_state.chat_log.append({"sender": "user", "message": user_input})
                    response = chatbot(user_input, user_lang_code)
                    st.session_state.chat_log.append({"sender": "bot", "message": response})
                    speak_response(response)

        if user_input:
            st.session_state.chat_log.append({"sender": "user", "message": user_input})
            response = chatbot(user_input, user_lang_code)
            st.session_state.chat_log.append({"sender": "bot", "message": response})
            speak_response(response)

    elif choice == "Conversation History":
        st.title("Conversation History")
        if len(st.session_state.chat_log) == 0:
            st.write("No conversation history yet.")
        else:
            for chat in st.session_state.chat_log:
                if chat['sender'] == 'user':
                    st.markdown(f"You: {chat['message']}")
                else:
                    st.markdown(f"Bot: {chat['message']}")

    elif choice == "About":
        st.title("About the Chatbot")
        st.write("""
        This chatbot assists users in preparing for job interviews by providing tips, answering common questions, 
        and offering advice. It utilizes NLP, machine learning, and multilingual support for an interactive experience.
        """)


if __name__ == '__main__':
    main()

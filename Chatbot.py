import os
import json
import random
import ssl
import nltk
import streamlit as st
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


# Main Application


# Main Application
def main():
    st.sidebar.image("chatbot_logo.png", width=150)
    menu_options = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Menu", menu_options)

    # Initialize session state variables
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""  # Temporary input storage

    if "process_input" not in st.session_state:
        st.session_state.process_input = False  # Flag to process input

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

        # Display chat log
        for chat in st.session_state.chat_log:
            if chat['sender'] == 'user':
                st.markdown(
                    f"<div style='padding:10px; background-color:#262730; border-radius:10px;color:#edf0eb;margin-bottom:5px;'><B>You:</B> {chat['message']}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='padding:10px; background-color:#E5E5E5; border-radius:10px;color:#020500; margin-bottom:5px;'><B>Bot:</B> {chat['message']}</div>",
                    unsafe_allow_html=True)

        # Input section
        col1, col2 = st.columns([8, 2])
        with col1:
            # Input field
            user_input = st.text_input(
                "Type your message:",
                value=st.session_state.user_input,
                key="user_input_home",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("Speak"):
                st.error("Features are temporarily unavailable.")

        # Check if input is provided
        if user_input and not st.session_state.process_input:
            st.session_state.user_input = user_input  # Save user input
            st.session_state.process_input = True  # Set flag to process input
            st.experimental_rerun()  # Rerun the app to handle input

        # Process input after rerun
        if st.session_state.process_input:
            # Add user's message to the chat log
            st.session_state.chat_log.append({"sender": "user", "message": st.session_state.user_input})

            # Generate bot response
            response = chatbot(st.session_state.user_input, user_lang_code)

            # Add bot's response to the chat log
            st.session_state.chat_log.append({"sender": "bot", "message": response})

            # Clear the input and reset flags
            st.session_state.user_input = ""  # Clear the input field
            st.session_state.process_input = False  # Reset processing flag
            st.experimental_rerun()  # Rerun to reflect cleared input

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

        # Display link to README file
        readme_file_path = "README.md"
        if os.path.exists(readme_file_path):
            with open(readme_file_path, "r") as file:
                readme_content = file.read()
            # Render the content of the README file
            st.markdown(readme_content)
        else:
            st.write("README.md file not found.")

if __name__ == '__main__':
    main()





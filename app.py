from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

from audio_recorder_streamlit import audio_recorder


# 利用可能なモデルと音声のリスト
MODEL_NAMES = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Streamlitのselectboxを使用してユーザーが選択できるようにする
selected_model_name = st.selectbox("モデルを選択してください", MODEL_NAMES)
selected_voice = st.selectbox("音声を選択してください", VOICES)


client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))


class ChatBot:
    def __init__(self, client, model_name, system_message, max_input_history=2):
        self.client = client
        self.model_name = model_name
        self.system_message = {"role": "system", "content": system_message}
        self.input_message_list = [self.system_message]
        self.max_input_history = max_input_history

    def add_user_message(self, message):
        self.input_message_list.append({"role": "user", "content": message})

    def get_ai_response(self, user_message):
        self.add_user_message(user_message)
        user_and_assisntant_message = self.input_message_list[1:]
        input_message_history = [self.system_message] + user_and_assisntant_message[
            -2 * self.max_input_history + 1 :
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=input_message_history,
            temperature=0,
        )
        ai_response = response.choices[0].message.content
        self.input_message_list.append({"role": "assistant", "content": ai_response})
        return ai_response


def initialize_chatbot(client, user_input):
    if "chatbot" not in st.session_state or st.session_state.user_input != user_input:
        st.session_state.chatbot = ChatBot(
            client,
            model_name=selected_model_name,
            system_message=user_input,
            max_input_history=5,
        )
        st.session_state.user_input = user_input
    return st.session_state.chatbot


def read_audio_file(file_path):
    with open(file_path, "rb") as audio_file:
        return audio_file.read()


def write_audio_file(file_path, audio_bytes):
    with open(file_path, "wb") as audio_file:
        audio_file.write(audio_bytes)


if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_input("system promptを設定してください", value=st.session_state.user_input)


if user_input:
    chatbot = initialize_chatbot(client, user_input)
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        write_audio_file("recorded_audio.wav", audio_bytes)

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=open("recorded_audio.wav", "rb"),
        )
        st.text(transcript.text)

        response_chatgpt = chatbot.get_ai_response(transcript.text)
        response = client.audio.speech.create(
            model="tts-1", voice=selected_voice, input=response_chatgpt
        )

        response.stream_to_file("speech.mp3")
        st.audio(read_audio_file("speech.mp3"), format="audio/mp3")

import base64
import logging
import os
import queue
import sys
import tempfile
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor

# import matplotlib.pyplot as plt
import numpy as np
import openai

# import pyaudio
import pydub
import requests
import streamlit as st
import yaml
from streamlit_webrtc import WebRtcMode
from streamlit_webrtc import webrtc_streamer

from chatbot_utils import run_audio_inference
from chatbot_utils import run_tts_inference

APP_NAME = "ChatGPT"


def init_states(local_session, config):
    local_session["API_KEY"] = local_session.get("API_KEY", "")
    local_session["START_SEGMENT"] = local_session.get("START_SEGMENT", "hello")
    local_session["STOP_SEGMENT"] = local_session.get("STOP_SEGMENT", "stop query")
    local_session["SKIP_SEGMENT"] = local_session.get("SKIP_SEGMENT", "skip query")
    local_session["CHATGPT_ROLE"] = local_session.get(
        "CHATGPT_ROLE", "You're a helpful and passionate assistant."
    )
    local_session["CHATBOX_TEXT"] = local_session.get("CHATBOX_TEXT", "")
    local_session["DETECTING_TEXT"] = local_session.get("DETECTING_TEXT", "")
    local_session["DETECTING_TEXT_CHANGED"] = local_session.get(
        "DETECTING_TEXT_CHANGED", False
    )
    local_session["CONN_INITIATED"] = local_session.get("CONN_INITIATED", False)
    local_session["USE_SILENCE_STOP"] = local_session.get("USE_SILENCE_STOP", False)
    local_session["AMOUNT_SILENCE_STOP_SECS"] = local_session.get(
        "AMOUNT_SILENCE_STOP_SECS", 3
    )

    local_session["STATUS_BOX_TEXT"] = local_session.get("STATUS_BOX_TEXT", "")
    local_session["STATUS_BOX_TEXT_CHANGED"] = local_session.get(
        "STATUS_BOX_TEXT_CHANGED", False
    )
    local_session["SESSION_STARTED"] = local_session.get("SESSION_STARTED", False)
    local_session["TO_CALL_TRANSCRIPTION"] = local_session.get(
        "TO_CALL_TRANSCRIPTION", False
    )
    local_session["TO_CALL_CHATGPT"] = local_session.get("TO_CALL_CHATGPT", False)
    local_session["CHAT_CONTEXT"] = local_session.get("CHAT_CONTEXT", [])
    local_session["TO_CALL_VOICEGEN"] = local_session.get("TO_CALL_VOICEGEN", False)
    local_session["TO_CALL_VOICEGEN_TEXT"] = local_session.get(
        "TO_CALL_VOICEGEN_TEXT", ""
    )

    local_session["PLAY_SOUND"] = local_session.get("PLAY_SOUND", False)
    local_session["PLAY_SOUND_PATH"] = local_session.get("PLAY_SOUND_PATH", "")

    local_session["GUI_DISABLED"] = local_session.get("GUI_DISABLED", False)
    local_session["CHATGPT_RANDOMNESS"] = local_session.get("CHATGPT_RANDOMNESS", 0.5)


def check_api_key(local_session, config):
    if config["API_KEY"] == "":
        return True


def play_sound(local_session, hiddenvoiceboxes, hiddenvoiceboxes_idx):
    with hiddenvoiceboxes[hiddenvoiceboxes_idx].empty():
        with open(local_session["PLAY_SOUND_PATH"], "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                    <div align="center">
                    <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    </div>
                    """
            hiddenvoiceboxes[hiddenvoiceboxes_idx].markdown(
                md,
                unsafe_allow_html=True,
            )
            # time.sleep(1)

    for idx, voicebox in enumerate(hiddenvoiceboxes):
        if idx != hiddenvoiceboxes_idx:
            voicebox.empty()

    hiddenvoiceboxes_idx = (hiddenvoiceboxes_idx + 1) % len(hiddenvoiceboxes)

    local_session["PLAY_SOUND"] = False

    return hiddenvoiceboxes_idx


def call_voice_gen(local_session):
    voice_file = tempfile.NamedTemporaryFile(suffix=".ogg")

    text2gen = local_session["TO_CALL_VOICEGEN_TEXT"]
    texts = text2gen.split("\n")

    _texts = []
    for text in texts:
        if text.strip() == "":
            continue
        _texts.append(text)
    texts = _texts

    def _get_audio_data(_text):
        audio_data = run_tts_inference(_text, config["NEMO_URL"])
        return audio_data

    generated_audio = pydub.AudioSegment.empty()

    with ThreadPoolExecutor(max_workers=4) as executor:
        audio_data_list = executor.map(_get_audio_data, texts)
        for audio_data in audio_data_list:
            with open(voice_file.name, "wb") as f:
                f.write(audio_data)

            # load audio
            audio = pydub.AudioSegment.from_file(voice_file.name, format="ogg")
            generated_audio += audio + pydub.AudioSegment.silent(duration=250)

    generated_audio.export(voice_file.name, format="mp3")

    local_session["PLAY_SOUND"] = True
    local_session["PLAY_SOUND_PATH"] = voice_file.name

    local_session["TO_CALL_VOICEGEN"] = False

    local_session[
        "STATUS_BOX_TEXT"
    ] = "Status: Connected!!! Waiting for '{}' to enter conversation session...".format(
        local_session["START_SEGMENT"]
    )
    local_session["STATUS_BOX_TEXT_CHANGED"] = True

    local_session["last_voice_file"] = voice_file


def call_chatgpt(local_session, config, chatbox):
    messages = [
        {
            "role": "system",
            "content": str(st.session_state["CHATGPT_ROLE"]),
        }
    ]

    for old_prompt, old_response in local_session["CHAT_CONTEXT"]:
        messages.append(
            {
                "role": "user",
                "content": old_prompt,
            },
        )
        messages.append(
            {
                "role": "assistant",
                "content": old_response,
            },
        )

    messages.append(
        {
            "role": "user",
            "content": local_session["TO_CALL_CHATGPT_PROMPT"],
        },
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=local_session["CHATGPT_RANDOMNESS"],
        max_tokens=config["CHATGPT_MAX_TOKENS"],
    )

    chatgpt_response = completion.choices[0].message.content

    while len(chatgpt_response) > 0 and chatgpt_response.startswith("\n"):
        chatgpt_response = chatgpt_response[1:]
    while len(chatgpt_response) > 0 and chatgpt_response.endswith("\n"):
        chatgpt_response = chatgpt_response[:-1]

    local_session["CHAT_CONTEXT"].append(
        (local_session["TO_CALL_CHATGPT_PROMPT"], chatgpt_response)
    )
    if len(local_session["CHAT_CONTEXT"]) > config["CHATGPT_CONTEXT_LENGTH"]:
        local_session["CHAT_CONTEXT"].pop(0)

    local_session["CHATBOX_TEXT"] = (
        "[ChatGPT]: " + chatgpt_response + "\n\n" + local_session["CHATBOX_TEXT"]
    )
    chatbox.text_area(
        "Chat log: ",
        local_session["CHATBOX_TEXT"],
        height=500,
    )

    # reset state
    local_session["TO_CALL_CHATGPT"] = False
    local_session["TO_CALL_CHATGPT_PROMPT"] = ""

    local_session["STATUS_BOX_TEXT"] = "Status: Generating chatgpt voice..."
    local_session["STATUS_BOX_TEXT_CHANGED"] = True

    local_session["TO_CALL_VOICEGEN"] = True
    local_session["TO_CALL_VOICEGEN_TEXT"] = chatgpt_response


def call_audio_transcription(local_session, chatbox, audio_to_transcribe):
    local_session["TO_CALL_TRANSCRIPTION"] = False

    user_voice_file = tempfile.NamedTemporaryFile(suffix=".mp3")

    audio_to_transcribe.export(user_voice_file.name, format="mp3")

    prompt = (
        run_audio_inference(user_voice_file.name, config["WHISPER_URL"]).strip().lower()
    )

    try:
        idx = prompt.index(local_session["START_SEGMENT"])
        prompt = prompt[idx + len(local_session["START_SEGMENT"]) :]
    except:
        pass

    try:
        idx = prompt.index(local_session["STOP_SEGMENT"])
        prompt = prompt[:idx]
    except:
        pass

    local_session["CHATBOX_TEXT"] = (
        "[YOU]: " + prompt + "\n\n" + local_session["CHATBOX_TEXT"]
    )
    chatbox.text_area(
        "Chat log: ",
        local_session["CHATBOX_TEXT"],
        height=500,
    )

    local_session["TO_CALL_CHATGPT"] = True
    local_session["TO_CALL_CHATGPT_PROMPT"] = prompt

    local_session["STATUS_BOX_TEXT"] = "Status: Waiting for ChatGPT to response..."
    local_session["STATUS_BOX_TEXT_CHANGED"] = True


def on_detected_text_changed(local_session, text_detection_area, audio_to_transcribe):
    text_detection_area.info(
        "Detecting text (last {:.1f} secs): {}".format(
            config["TMP_AUDIO_BUFFER_LENGTH"] / 1000.0,
            local_session["DETECTING_TEXT"],
        )
    )
    local_session["DETECTING_TEXT_CHANGED"] = False

    text = local_session["DETECTING_TEXT"].lower()

    # check if hotword was detected
    if (
        local_session["START_SEGMENT"] in text
        and local_session["SESSION_STARTED"] is False
    ):
        # start session
        local_session["SESSION_STARTED"] = True
        local_session["SESSION_AUDIO_BUFFER"] = pydub.AudioSegment.silent(
            duration=config["SESSION_AUDIO_BUFFER_LENGTH"]
        )
        local_session["SESSION_AUDIO_BUFFER"] += local_session["sound_window_buffer"]
        local_session["SESSION_AUDIO_BUFFER"] = local_session["SESSION_AUDIO_BUFFER"][
            -config["SESSION_AUDIO_BUFFER_LENGTH"] :
        ]

        # set status
        local_session[
            "STATUS_BOX_TEXT"
        ] = "Status: Conversation session started. Say '{}' to receive response from ChatGPT or {} to cancel this session.".format(
            local_session["STOP_SEGMENT"], local_session["SKIP_SEGMENT"]
        )
        local_session["STATUS_BOX_TEXT_CHANGED"] = True
        local_session["PLAY_SOUND"] = True
        local_session["PLAY_SOUND_PATH"] = "data/session_started.ogg"
    elif local_session["SKIP_SEGMENT"] in text and local_session["SESSION_STARTED"]:
        # ignore this session
        local_session["SESSION_STARTED"] = False
        local_session[
            "STATUS_BOX_TEXT"
        ] = "Status: Connected!!! Waiting for '{}' to enter conversation session...".format(
            local_session["START_SEGMENT"]
        )
        local_session["STATUS_BOX_TEXT_CHANGED"] = True
        local_session["PLAY_SOUND"] = True
        local_session["PLAY_SOUND_PATH"] = "data/session_cancelled.ogg"
    elif local_session["STOP_SEGMENT"] in text and local_session["SESSION_STARTED"]:
        local_session["SESSION_STARTED"] = False
        local_session[
            "STATUS_BOX_TEXT"
        ] = "Status: Transcribing your voice into text..."
        local_session["TO_CALL_TRANSCRIPTION"] = True
        local_session["STATUS_BOX_TEXT_CHANGED"] = True
        audio_to_transcribe = local_session["SESSION_AUDIO_BUFFER"]
        local_session["PLAY_SOUND"] = True
        local_session["PLAY_SOUND_PATH"] = "data/session_end.ogg"

    return audio_to_transcribe


def main(config):
    logger = logging.getLogger(__name__)

    st.title("ChatGPT Audio Bot")

    # launch webrtc stream
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=config["WEBRTC_AUDIO_BUFFER_SIZE"],
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                        "stun:stun2.l.google.com:19302",
                        "stun:stun3.l.google.com:19302",
                        "stun:stun4.l.google.com:19302",
                    ]
                }
            ]
        },
        media_stream_constraints={"audio": True},
    )

    if APP_NAME not in st.session_state:
        st.session_state[APP_NAME] = {}

    local_session = st.session_state[APP_NAME]

    init_states(local_session, config)

    if config["API_KEY"] != "":
        local_session["API_KEY"] = st.text_input(
            "API key:", value=local_session["API_KEY"], type="password"
        )

        api_key_area = st.empty()
        if local_session["API_KEY"] != config["API_KEY"]:
            api_key_area.error("API key is incorrect")

    # setup start/stop/skip words
    st.text(
        "Say '{}'/'{}'/'{}' to start/stop/skip conversation session.".format(
            local_session["START_SEGMENT"],
            local_session["STOP_SEGMENT"],
            local_session["SKIP_SEGMENT"],
        )
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        local_session["START_SEGMENT"] = st.text_input(
            "Start session:",
            value=local_session["START_SEGMENT"],
            disabled=local_session["GUI_DISABLED"],
        ).lower()
    with col2:
        local_session["STOP_SEGMENT"] = st.text_input(
            "Stop session:",
            value=local_session["STOP_SEGMENT"],
            disabled=local_session["GUI_DISABLED"],
        ).lower()

        use_silent_stop = local_session["USE_SILENCE_STOP"]
        local_session["USE_SILENCE_STOP"] = st.checkbox(
            "Silent to stop",
            value=use_silent_stop,
            disabled=local_session["GUI_DISABLED"],
        )
        if local_session["USE_SILENCE_STOP"]:
            amount_to_stop = local_session["AMOUNT_SILENCE_STOP_SECS"]
            local_session["AMOUNT_SILENCE_STOP_SECS"] = st.slider(
                "Silence time (s)",
                min_value=2,
                max_value=10,
                value=amount_to_stop,
                disabled=local_session["GUI_DISABLED"],
            )

    with col3:
        local_session["SKIP_SEGMENT"] = st.text_input(
            "Skip session:",
            value=local_session["SKIP_SEGMENT"],
            disabled=local_session["GUI_DISABLED"],
        ).lower()

    rolebox = st.empty()
    local_session["CHATGPT_ROLE"] = rolebox.text_area(
        "ChatGPT role:",
        value=local_session["CHATGPT_ROLE"],
        key="CHATGPT_ROLE",
        max_chars=150,
        height=10,
        disabled=local_session["GUI_DISABLED"],
    )

    local_session["CHATGPT_RANDOMNESS"] = st.slider(
        "ChatGPT randomness",
        min_value=0.0,
        max_value=2.0,
        value=local_session["CHATGPT_RANDOMNESS"],
        step=0.1,
        disabled=local_session["GUI_DISABLED"],
    )

    statusbox = st.empty()
    statusbox.info(
        'Status: Choose your audio source, configure all settings and click "Start" to connect...'
    )

    text_detection_area = st.empty()
    # text_detection_area.info("Detecting text: ")

    chatbox = st.empty()
    chatbox.text_area(
        "Chat log: ",
        local_session["CHATBOX_TEXT"],
        height=500,
    )

    # WORKAROUND: use multiple containers to play audio because sometimes the audio is not updated on GUI
    hiddenvoiceboxes = [st.empty() for _ in range(20)]
    hiddenvoiceboxes_idx = 0

    if "sound_window_buffer" not in local_session:
        local_session["sound_window_buffer"] = pydub.AudioSegment.silent(
            duration=config["TMP_AUDIO_BUFFER_LENGTH"]
        )

    audio_to_transcribe = None
    conn_init = False
    last_check_hotword_ts = time.time()
    last_silent_ts = time.time()

    while local_session["API_KEY"] == config["API_KEY"]:
        if local_session["PLAY_SOUND"]:
            hiddenvoiceboxes_idx = play_sound(
                local_session, hiddenvoiceboxes, hiddenvoiceboxes_idx
            )

        if local_session["TO_CALL_VOICEGEN"]:
            call_voice_gen(local_session)

        if local_session["TO_CALL_CHATGPT"]:
            call_chatgpt(local_session, config, chatbox)

        if local_session["TO_CALL_TRANSCRIPTION"]:
            call_audio_transcription(local_session, chatbox, audio_to_transcribe)
            audio_to_transcribe = None

        if local_session["STATUS_BOX_TEXT_CHANGED"]:
            statusbox.empty()
            statusbox.info(local_session["STATUS_BOX_TEXT"], icon="🤖")
            local_session["STATUS_BOX_TEXT_CHANGED"] = False

        if local_session["DETECTING_TEXT_CHANGED"]:
            audio_to_transcribe = on_detected_text_changed(
                local_session, text_detection_area, audio_to_transcribe
            )

        if local_session["USE_SILENCE_STOP"] and local_session["SESSION_STARTED"]:
            # special case: if no sound is detected for a long time, stop the session
            if time.time() - last_silent_ts > local_session["AMOUNT_SILENCE_STOP_SECS"]:
                local_session["SESSION_STARTED"] = False
                local_session[
                    "STATUS_BOX_TEXT"
                ] = "Status: Transcribing your voice into text..."
                local_session["TO_CALL_TRANSCRIPTION"] = True
                local_session["STATUS_BOX_TEXT_CHANGED"] = True
                audio_to_transcribe = local_session["SESSION_AUDIO_BUFFER"]
                local_session["PLAY_SOUND"] = True
                local_session["PLAY_SOUND_PATH"] = "data/session_end.ogg"

        if not conn_init:
            # FIXME: this is a workaround to wait untit webrtc connection is established
            if not local_session["CONN_INITIATED"]:
                local_session["STATUS_BOX_TEXT"] = "Status: Waiting for connection..."
                local_session["STATUS_BOX_TEXT_CHANGED"] = True
            else:
                local_session[
                    "STATUS_BOX_TEXT"
                ] = "Status: Connected!!! Waiting for '{}' to enter conversation session...".format(
                    local_session["START_SEGMENT"]
                )
                local_session["STATUS_BOX_TEXT_CHANGED"] = True
                conn_init = True

        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                if not local_session["CONN_INITIATED"]:
                    local_session["CONN_INITIATED"] = True
                if not local_session["GUI_DISABLED"]:
                    local_session["GUI_DISABLED"] = True
                    # force restart once
                    break

            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                time.sleep(0.1)
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                local_session["sound_window_buffer"] += sound_chunk

                # add chunk to session buffer if session is started
                if local_session["SESSION_STARTED"]:
                    local_session["SESSION_AUDIO_BUFFER"] += sound_chunk
                    local_session["SESSION_AUDIO_BUFFER"] = local_session[
                        "SESSION_AUDIO_BUFFER"
                    ][-config["SESSION_AUDIO_BUFFER_LENGTH"] :]

                # trim recording buffer
                if (
                    len(local_session["sound_window_buffer"])
                    > config["TMP_AUDIO_BUFFER_LENGTH"]
                ):
                    local_session["sound_window_buffer"] = local_session[
                        "sound_window_buffer"
                    ][-config["TMP_AUDIO_BUFFER_LENGTH"] :]

                if (
                    time.time() - last_check_hotword_ts
                    > config["HOTWORD_DETECTION_INTERVAL_MS"] / 1000.0
                ):
                    # detect non silience
                    results = pydub.silence.detect_nonsilent(
                        local_session["sound_window_buffer"],
                        min_silence_len=config["PYDUB_MIN_SILENCE_LEN"],
                        silence_thresh=config["PYDUB_SILENCE_THRESHOLD"],
                    )

                    # if len(results) == 0:
                    #     print("Silence detected...")
                    #     print(
                    #         "Time since last silent: {}".format(
                    #             time.time() - last_silent_ts
                    #         )
                    #     )

                    if len(results) > 0:
                        last_silent_ts = time.time()

                        tmp_audio_window_file = tempfile.NamedTemporaryFile(
                            suffix=".mp3"
                        )

                        local_session["sound_window_buffer"].export(
                            tmp_audio_window_file.name, format="mp3"
                        )

                        text = (
                            run_audio_inference(
                                tmp_audio_window_file.name, config["WHISPER_URL"]
                            )
                            .strip()
                            .lower()
                        )

                        if text != "":
                            local_session["DETECTING_TEXT"] = text
                            local_session["DETECTING_TEXT_CHANGED"] = True

                    last_check_hotword_ts = time.time()
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))

    # override config with env variables
    if "OPENAI_API_KEY" in os.environ:
        config["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

    if "CHATGPT_CONTEXT_LENGTH" in os.environ:
        config["CHATGPT_CONTEXT_LENGTH"] = int(os.environ["CHATGPT_CONTEXT_LENGTH"])

    if "CHATGPT_MAX_TOKENS" in os.environ:
        config["CHATGPT_MAX_TOKENS"] = int(os.environ["CHATGPT_MAX_TOKENS"])

    if "API_KEY" in os.environ:
        config["API_KEY"] = os.environ["API_KEY"]

    assert config["OPENAI_API_KEY"] != "", "OPENAI_API_KEY is not set"

    openai.api_key = config["OPENAI_API_KEY"]
    main(config)

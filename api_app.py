# create default fastapi config
import argparse
import os
import tempfile
import time
import uuid
from threading import Lock

import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.models.base import Vocoder
from omegaconf import OmegaConf
from omegaconf import open_dict

from api_app_utils import generate_audio
from api_app_utils import load_spectrogram_model
from api_app_utils import load_vocoder_model
from whisperX import whisperx

mutex = None
device = "cuda"
whisper_model = None
nemo_model = None

spectrogram_generator = "mixertts"
audio_generator = "univnet"


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=10000)
parser.add_argument("--model_name", type=str, default="medium.en")
args = parser.parse_args()


def load_model():
    global whisper_model, nemo_model, mutex

    if mutex is None:
        mutex = Lock()

    if whisper_model is None:
        mutex = Lock()
        whisper_model = whisperx.load_model(args.model_name, device)

    if nemo_model is None:
        spec_gen = load_spectrogram_model("mixertts").eval().cuda()
        vocoder = load_vocoder_model("univnet").eval().cuda()
        nemo_model = (spec_gen, vocoder)


app = FastAPI(
    title="Text transcription and Audio generation FastAPI",
    version="0.1.0",
    contact={"name": "Lanytek", "email": "lanytek@gmail.com"},
)


@app.get("/")
def load_all_models():
    global whisper_model, mutex
    load_model()

    return {}


@app.post("/api/v1/audios/transcribe")
async def transcribe_audio_to_text(audio_file: UploadFile = File(...)):
    global whisper_model, mutex
    load_model()

    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, "wb") as f:
        data = audio_file.file.read()
        f.write(data)

    with mutex:
        result = whisper_model.transcribe(tmp_file.name)

    return {"text": result["text"]}


@app.post("/api/v1/audios/generate")
async def generate_audio_from_text(text: str):
    global nemo_model, mutex
    load_model()

    with mutex:
        _, audio = generate_audio(
            spectrogram_generator, nemo_model[0], audio_generator, nemo_model[1], text
        )

    tmpfile = tempfile.NamedTemporaryFile(suffix=".ogg")
    sf.write(tmpfile.name, audio[0], 22050)
    data = open(tmpfile.name, "rb").read()

    return Response(content=data, media_type="audio/ogg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_app:app", host="0.0.0.0", port=args.port, reload=False)

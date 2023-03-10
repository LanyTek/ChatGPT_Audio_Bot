import requests


def run_audio_inference(audio_file, whisper_url):
    """_summary_

    Args:
        audio_file (_type_): _description_
        whisper_url (_type_): _description_

    Returns:
        _type_: _description_
    """

    return requests.post(
        f"{whisper_url}/api/v1/audios/transcribe",
        files={"audio_file": open(audio_file, "rb").read()},
        timeout=10000,
    ).json()["text"]


def run_tts_inference(text, nemo_url):
    """_summary_

    Args:
        text (_type_): _description_
        nemo_url (_type_): _description_

    Returns:
        _type_: _description_
    """

    return requests.post(
        f"{nemo_url}/api/v1/audios/generate",
        params={
            "text": text,
        },
        timeout=10000,
    ).content

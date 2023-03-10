import torch
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.models.base import Vocoder


def load_spectrogram_model(spectrogram_generator):
    override_conf = None

    from_pretrained_call = SpectrogramGenerator.from_pretrained

    if spectrogram_generator == "tacotron2":
        from nemo.collections.tts.models import Tacotron2Model

        pretrained_model = "tts_en_tacotron2"
    elif spectrogram_generator == "fastpitch":
        from nemo.collections.tts.models import FastPitchModel

        pretrained_model = "tts_en_fastpitch"
    elif spectrogram_generator == "mixertts":
        from nemo.collections.tts.models import MixerTTSModel

        pretrained_model = "tts_en_lj_mixertts"
    elif spectrogram_generator == "mixerttsx":
        from nemo.collections.tts.models import MixerTTSModel

        pretrained_model = "tts_en_lj_mixerttsx"
    else:
        raise NotImplementedError

    return from_pretrained_call(pretrained_model, override_config_path=override_conf)


def load_vocoder_model(audio_generator):
    TwoStagesModel = False
    strict = True

    if audio_generator == "waveglow":
        from nemo.collections.tts.models import WaveGlowModel

        pretrained_model = "tts_waveglow"
        strict = False
    elif audio_generator == "hifigan":
        from nemo.collections.tts.models import HifiGanModel

        spectrogram_generator2ft_hifigan = {
            "mixertts": "tts_en_lj_hifigan_ft_mixertts",
            "mixerttsx": "tts_en_lj_hifigan_ft_mixerttsx",
        }
        pretrained_model = spectrogram_generator2ft_hifigan.get(
            spectrogram_generator, "tts_hifigan"
        )
    elif audio_generator == "univnet":
        from nemo.collections.tts.models import UnivNetModel

        pretrained_model = "tts_en_lj_univnet"
    elif audio_generator == "griffin-lim":
        from nemo.collections.tts.models import TwoStagesModel

        cfg = {
            "linvocoder": {
                "_target_": "nemo.collections.tts.models.two_stages.GriffinLimModel",
                "cfg": {"n_iters": 64, "n_fft": 1024, "l_hop": 256},
            },
            "mel2spec": {
                "_target_": "nemo.collections.tts.models.two_stages.MelPsuedoInverseModel",
                "cfg": {
                    "sampling_rate": 22050,
                    "n_fft": 1024,
                    "mel_fmin": 0,
                    "mel_fmax": 8000,
                    "mel_freq": 80,
                },
            },
        }
        _model = TwoStagesModel(cfg)
        TwoStagesModel = True
    else:
        raise NotImplementedError

    if not TwoStagesModel:
        _model = Vocoder.from_pretrained(pretrained_model, strict=strict)

    return _model


def generate_audio(
    spectrogram_generator, spec_gen_model, audio_generator, vocoder_model, str_input
):
    with torch.no_grad():
        parsed = spec_gen_model.parse(str_input)
        gen_spec_kwargs = {}

        if spectrogram_generator == "mixerttsx":
            gen_spec_kwargs["raw_texts"] = [str_input]

        spectrogram = spec_gen_model.generate_spectrogram(
            tokens=parsed, **gen_spec_kwargs
        )
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)

        if audio_generator == "hifigan":
            audio = vocoder_model._bias_denoise(audio, spectrogram).squeeze(1)

    if spectrogram is not None:
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.to("cpu").numpy()
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.to("cpu").numpy()
    return spectrogram, audio

import torch
import numpy as np
from librosa import load, resample, stream, get_samplerate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from spellchecker import SpellChecker
from tqdm.notebook import tqdm


class Transcriptor:
    def __init__(self, model_name=None, spell_checker=True):
        if model_name is not None:
            self.model_name = model_name
            self.processor_name = model_name

        else:
            self.model_name = 'dannersm/wav2vec2-large-xlsr-53-chilean-lessons'
            self.processor_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self._processor = Wav2Vec2Processor.from_pretrained(self.processor_name)

        self.spell_checker = None
        if spell_checker:
            self.spell_checker = SpellChecker(language='es')

    # transcribes a segment using a fine tuned Wav2Vec2ForCTC from huggingface transformer's framework
    def _transcribe_segment(self, y, sr):
        # check for device to cast input tensor accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = self._processor(resample(y, sr, 16_000), sampling_rate=16_000, return_tensors='pt', padding=True).to(
            device)

        # predict and decode accordingly,
        # we use batch decode since the regular decode works strangely with Wav2Vec2 models
        with torch.no_grad():
            predicted_ids = torch.argmax(self._model(inputs.input_values, attention_mask=inputs.attention_mask).logits,
                                         dim=-1)
        return self._processor.batch_decode(predicted_ids)[0]

    # transcribes a whole file and outputs a string
    def _transcribe_full_file(self, audio_file, offset=0.0, duration=None):
        y, sr = load(audio_file, sr=None, offset=offset, duration=duration)
        return self._transcribe_segment(y, sr)

    # transcribes by streaming the audio every "interval" seconds and returns a list of strings
    def _transcribe_in_stream(self, audio_file, interval, offset=0.0, duration=None, n_frames=100, hop_length=100):
        sr = get_samplerate(audio_file)
        audio_stream = stream(audio_file, block_length=np.ceil(sr * interval / n_frames).astype(int),
                              offset=offset,
                              duration=duration,
                              frame_length=n_frames,
                              hop_length=hop_length)

        return [self._transcribe_segment(y, sr) for y in tqdm(audio_stream)]

    # uses spanish version of pyspellchecker to correct spelling
    def _check_spelling(self, text):
        if self.spell_checker:
            return ' '.join([self.spell_checker.correction(w) for w in text.split()])
        else:
            return text

    def transcribe(self, audio_file, interval=10, offset=0.0, duration=None, n_frames=100, hop_length=100):
        # check device and move model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(device)

        # transcribe according to interval or not
        if interval:
            return [self._check_spelling(line).lower() for line
                    in self._transcribe_in_stream(audio_file, interval, offset, duration, n_frames, hop_length)]
        else:
            return self._check_spelling(self._transcribe_at_once(audio_file, offset, duration)).lower()

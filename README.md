# CLTranscriptor

Wrapper for spanish speech-to-text model based on huggingface's [Wav2Vec2ForCTC](https://huggingface.co/transformers/model_doc/wav2vec2.html) finetuned on Chilean lessons + [PySpellChecker](https://pyspellchecker.readthedocs.io/en/latest/)'s spanish spellchecking algorithm.

## Install
To install, simply use `pip`:

```python
!pip install cltranscriptor
```
## Usage
To use, initialize a `Transcriptor` object:

```python
from cltranscriptor.cltranscriptor import Transcriptor
transcriptor = Transcriptor()
```
By default, spell checking is set to `True` and the model name is the one available at [dannersm/wav2vec2-large-xlsr-53-chilean-lessons](https://huggingface.co/dannersm/wav2vec2-large-xlsr-53-chilean-lessons), which is based on Jonatas Grosman's [model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish) and finetuned on a 6 hour set of chilean lessons.

To transcribe a file, call `Transcriptor.transcribe()`:
```python
transcriptor.transcribe('/path/to/your/audio_file.wav')
```
By default, the file is streamed into 10 second intervals (to avoid loading it in memory) and returns a list with the transcripts for each segment. If you want to transcribe a *relatively short* file all at once you can pass `interval=None`:
```python
transcriptor.transcribe('my_file.wav', interval=None)
```
You can also pass the `offset` and `duration` parameters which will be passed to `librosa.stream` to set the start time and a maximum duration to the transcription
```python
transcriptor.transcribe('my_file.wav', offset=600, duration=120) # transcribe 2 minutes of audio starting from minute 10
```
Finally, you can control the length of the streamed segments passing `interval`:
```python
transcriptor.transcribe('my_file.wav', interval=15) # transcribe every 15 seconds 
```

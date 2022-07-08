import os
import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
import IPython
import sounddevice as sd
from scipy.io.wavfile import write

torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torchaudio.set_audio_backend("soundfile")

# print(torch.__version__)
# print(torchaudio.__version__)
# print(device)

SPEECH_FILE = "sound/speech.wav"

def recordAudio():
    fs = 16000  # Sample rate
    seconds = 13  # Duration of recording

    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished

    if not os.path.exists(SPEECH_FILE):
        os.makedirs('_assets', exist_ok=True)    
        write(SPEECH_FILE, fs, recording)  # Save as WAV file

recordAudio()

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
print("Sample Rate:", bundle.sample_rate)
print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)
print(model.__class__)

IPython.display.Audio(SPEECH_FILE)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)
sample_rate


if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode():
    features, _ = model.extract_features(waveform)

fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu())
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
plt.tight_layout()
#plt.savefig('acoustic_features.png')
plt.show()

with torch.inference_mode():
    emission, _ = model(waveform)

plt.imshow(emission[0].cpu().T)
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.show()
print("Class labels:", bundle.get_labels())


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, ignore):
        super().__init__()
        self.labels = labels
        self.ignore = ignore

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i not in self.ignore]
        return ''.join([self.labels[i] for i in indices])

greedy_decoder = GreedyCTCDecoder(
    labels=bundle.get_labels(),
    ignore=(0,0),
)
greedy_result = greedy_decoder(emission[0])
greedy_transcript = greedy_result.replace("|", " ").lower().strip()

IPython.display.Audio(SPEECH_FILE)

print(greedy_transcript)



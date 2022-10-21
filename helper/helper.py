import matplotlib.pyplot as plt
from matplotlib import cm
import pyaudio
import wave
import librosa
from pathlib import Path
#import sounddevice as sd
#import wavio
import numpy as np


def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    if ax is None:
        ax = plt.gca()
    
    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)
    
    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    mappable.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)


def draw_embed(embed, name, which):
    """
    Draws an embedding.
    Parameters:
        embed (np.array): array of embedding
        name (str): title of plot
    Return:
        fig: matplotlib figure
    """
    fig, embed_ax = plt.subplots()
    plot_embedding_as_heatmap(embed)
    embed_ax.set_title(name)
    embed_ax.set_aspect("equal", "datalim")
    embed_ax.set_xticks([])
    embed_ax.set_yticks([])
    embed_ax.figure.canvas.draw()
    return fig


def create_spectrogram(voice_sample):
    """
    Creates and saves a spectrogram plot for a sound sample.
    Parameters:
        voice_sample (str): path to sample of sound
    Return:
        fig
    """

    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure()
    plt.subplot(211)
    plt.title("Spectrogram of your sample")

    plt.plot(original_wav)
    #plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(212)
    plt.specgram(original_wav, Fs=sampling_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.savefig(voice_sample.split(".")[0] + "_spectogram.png")
    return fig

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

# def record(duration=5, fs=48000):
#     sd.default.samplerate = fs
#     sd.default.channels = 1
#     myrecording = sd.rec(int(duration * fs))
#     sd.wait(duration)
#     return myrecording



def record(duration=5, fs=44100):
	# start Recording
	audio = pyaudio.PyAudio()
	chunk = 1024
	stream = audio.open(
		format = pyaudio.paInt16,
		channels = 1,
		rate = fs,
		input=True,
		frames_per_buffer=chunk,
		input_device_index=1)
	frames = []
	for i in range(0, int(fs / chunk * duration)):
		data = stream.read(chunk)
		frames.append(data)
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
	return frames





def save_record(path_myrecording, frames, fs):
#   wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
	audio = pyaudio.PyAudio()
	format = pyaudio.paInt16
	waveFile = wave.open(path_myrecording, 'wb')
	waveFile.setnchannels(1)
	waveFile.setsampwidth(audio.get_sample_size(format))
	waveFile.setframerate(fs)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()
	return None
    
from flask import Flask, render_template, request, redirect, url_for,jsonify
import numpy as np
import librosa
import sounddevice as sd
from scipy.ndimage import maximum_filter1d
import matplotlib.pyplot as plt
import pyaudio
import wave
import os
from dtaidistance import dtw
app = Flask(__name__)

# 載入音檔
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
def record_audio_to_file(filename, duration=4, channels=1, rate=44100, frames_per_buffer=1024):
    """Record user's input audio and save it to the specified file."""
    FORMAT = pyaudio.paInt16
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT, channels=channels, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
    print("Start recording...")

    frames = []
    # Record for the given duration
    for _ in range(0, int(rate / frames_per_buffer * duration)):
        data = stream.read(frames_per_buffer)
        frames.append(data)

    print("Recording stopped")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio recorded and saved to {filename}")
# 提取 MFCC 特徵
def extract_mfcc(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs

# 正規化 MFCC 特徵
def normalize_mfcc(mfccs):
    return (mfccs - np.mean(mfccs)) / np.std(mfccs)

# 計算相似度分數
def compute_similarity_score(mfccs_A, mfccs_B):
    return np.mean(np.abs(mfccs_A - mfccs_B)) * 100

# 去除靜音部分
def remove_silence(audio, threshold=0.02):
    non_silent_intervals = librosa.effects.split(audio, top_db=threshold)
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return non_silent_audio

# 預處理雜音
def preprocess_audio(audio):
    # 雜音預處理，例如濾波等
    # 這裡我們使用 maximum_filter1d 函數來濾波
    filtered_audio = maximum_filter1d(audio, size=3)
    return filtered_audio
@app.route('/get_audio_length', methods=['POST'])
def get_audio_length():
    selected_file = request.json['speechFile']
    audio_file_path_A = os.path.join('C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\', selected_file)
    audio_A, sr_A = load_audio(audio_file_path_A)
    duration = librosa.get_duration(y=audio_A, sr=sr_A)
    return jsonify(length=duration)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    file.save(os.path.join('C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\user_input.wav'))
    return 'File uploaded successfully', 200
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_file = request.form['speechFile']
        # 載入音檔 A
        audio_file_path_A = os.path.join('C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\',selected_file)
        audio_A, sr_A = load_audio(audio_file_path_A)
        # 載入音檔 B
        audio_file_path_B = os.path.join('C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\user_input.wav')
        # record_audio_to_file(audio_file_path_B, duration=4, channels=1, rate=44100, frames_per_buffer=1024)
        audio_B, sr_B = load_audio(audio_file_path_B)
        # 去除靜音部分
        audio_A = remove_silence(audio_A)
        audio_B = remove_silence(audio_B)
        # 預處理雜音
        audio_preA = preprocess_audio(audio_A)
        audio_preB = preprocess_audio(audio_B)
        # 提取 MFCC 特徵
        mfccs_A = extract_mfcc(audio_A, sr_A)
        mfccs_B = extract_mfcc(audio_B, sr_A)
        # 使兩個音檔的 MFCC 特徵具有相同的維度
        min_length = min(mfccs_A.shape[1], mfccs_B.shape[1])
        mfccs_A = mfccs_A[:, :min_length]
        mfccs_B = mfccs_B[:, :min_length]
        # 正規化 MFCC 特徵
        mfccs_A_normalized = normalize_mfcc(mfccs_A)
        mfccs_B_normalized = normalize_mfcc(mfccs_B)
        # 計算相似度分數
        score = 100-compute_similarity_score(mfccs_A_normalized, mfccs_B_normalized)
        return render_template('index.html', score=score)
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()

    # audio_file_path_A = os.path.join('C:\\Users\\USER\\Desktop\\flask-templete\\templates\\A.wav')
    # audio_A, sr_A = load_audio(audio_file_path_A)
    # audio_file_path_B = os.path.join('C:\\Users\\USER\\Desktop\\flask-templete\\templates\\user_input.wav')
    # record_audio_to_file(audio_file_path_B, duration=4, channels=1, rate=44100, frames_per_buffer=1024)
    # audio_B, sr_B = load_audio(audio_file_path_B)
    # # 去除靜音部分
    # audio_A1 = remove_silence(audio_A)
    # audio_B1 = remove_silence(audio_B)
    # # 預處理雜音
    # audio_preA2 = preprocess_audio(audio_A1)
    # audio_preB2 = preprocess_audio(audio_B1)
    # # 提取 MFCC 特徵
    # mfccs_A = extract_mfcc(audio_A, sr_A)
    # mfccs_B = extract_mfcc(audio_B, sr_A)
    # # 使兩個音檔的 MFCC 特徵具有相同的維度
    # min_length = min(mfccs_A.shape[1], mfccs_B.shape[1])
    # mfccs_A = mfccs_A[:, :min_length]
    # mfccs_B = mfccs_B[:, :min_length]
    # # 正規化 MFCC 特徵
    # mfccs_A_normalized = normalize_mfcc(mfccs_A)
    # mfccs_B_normalized = normalize_mfcc(mfccs_B)
    # plt.figure()
    # librosa.display.waveshow(mfccs_A_normalized, sr=sr_A)
    # plt.title('waveform')
    # plt.show()
    # plt.figure()
    # librosa.display.waveshow(mfccs_B_normalized, sr=sr_B)
    # plt.title('waveform')
    # plt.show()
import wave
import pickle
import pyaudio
import librosa
import numpy as np
from sklearn import metrics
from librosa import feature
from collections import Counter


WINDOW_SIZE = 1024
CHANNELS = 1
RATE = 44100
FFT_FRAMES_IN_SPEC = 19

# global
global_blocks = np.zeros((FFT_FRAMES_IN_SPEC, WINDOW_SIZE))
fft_frame = np.array(WINDOW_SIZE // 2)
win = np.hamming(WINDOW_SIZE)
spec_img = np.zeros((WINDOW_SIZE // 2, FFT_FRAMES_IN_SPEC))


def load_sound_file_into_memory(path):
    audio_data = wave.open(path, 'rb')
    return audio_data


f = load_sound_file_into_memory('test_audio.wav')

# check how many channels - e.g., mono=1, stereo=2 etc.
# print('number of channels: ', f.getnchannels())
# check frame rate - e.g., 44100 (CD quality) etc
# print('sample rate: ', f.getframerate())
# check sample width in bytes, - e.g., 8-bit=1byte, 16-bit=2bytes (CD quality), etc
# print('sample width in bytes: ', f.getsampwidth())

# First, make sure the file-reading is in the beginning of the file
f.setpos(0)
# Let's read 10 samples
samples = f.readframes(10)
print('10 samples: ', samples)
# The array of 10 samples has actually 20 byte values, 2 bytes (16 bits) for each sample
# Numpy can convert the 20 bytes to an integer array, by (internally) taking
# the bytes in the correct order (according to little or big endian setup).
samples_np = np.frombuffer(samples, dtype='int16')
print('10 samples in numpy array: ', samples_np)

# Read one frame from the audio file
frames = f.readframes(1)


def callback( in_data, frame_count, time_info, status):
    global global_blocks, fft_frame, win, spec_img
    # begin with a zero buffer
    block_for_speakers = np.zeros( (WINDOW_SIZE , CHANNELS) , dtype='int16' )
    # get bytes from RAM, where the file is loaded as a continuous chunck
    block_bytes_from_file = f.readframes(WINDOW_SIZE)
    # transform block from bytes to numpy values - 16-bit audio
    # CAUTION: we need to know that the file is 16-bit audio!
    # to do it properly, for any bit depth, we need to rely on information
    # obtained from f.getsampwidth() - but it's too complicated for now...
    numpy_block_from_bytes = np.frombuffer( block_bytes_from_file , dtype='int16' )
    # 0 column is left, 1 is right speaker / channel
    block_for_speakers[:,0] += numpy_block_from_bytes
    # ... or both speakers
    #block_for_speakers[:,0] += numpy_block_from_bytes
    if len(win) == len(numpy_block_from_bytes):
        frame_fft = np.fft.fft(win * numpy_block_from_bytes)
        p = np.abs(frame_fft) * 2 / np.sum(win)
        # translate in dB
        log_items = p[: WINDOW_SIZE // 2] / 32678
        fft_frame = 20 * np.log10(log_items)
        spec_img = np.roll(spec_img, -1, axis=1)
        spec_img[:, -1] = fft_frame[::-1]
        global_blocks = np.roll(global_blocks, -1, axis=0)
        global_blocks[-1, :] = numpy_block_from_bytes
    return (block_for_speakers, pyaudio.paContinue)


p = pyaudio.PyAudio()
output = p.open(format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=WINDOW_SIZE,
            stream_callback=callback,
            start=False)

output.start_stream()

# Load the pickle model
try:
    with open('best_clf.pkl', 'rb') as file:
        model = pickle.load(file)
except EOFError:
    print("Error: Unable to load the pickle file. The file may be empty or corrupted.")


def majority_voting(x):
    counter = Counter(x)
    majority_element = max(counter, key=counter.get)
    return majority_element

predictions = []
for sample in samples:
    s = np.reshape(global_blocks, WINDOW_SIZE * FFT_FRAMES_IN_SPEC)
    sample_mfcc = librosa.feature.mfcc(y=s, sr=RATE, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE)
    # sample_mfcc = sample_mfcc[1:]
    # print(sample_mfcc)
    instance = sample_mfcc[0]
    sample_instances = [instance]
    sample_predictions = model.predict(sample_instances)
    predictions.append(sample_predictions[0])

    # print(predictions)
print(f'predictions are {predictions}', predictions)
majority_prediction = majority_voting(predictions)
print(f'majority is {majority_prediction}', majority_prediction)
# print(prediction)
if majority_prediction == 0.0:
    print('surprised')
else:
    print('calm')

prediction_array = [majority_prediction]
true_label = [0.0]

# Calculating Accuracy
print("Accuracy (in %):", metrics.accuracy_score(true_label, prediction_array) * 100)
# CAUTION: output stream needs to stop before starting it again!
output.stop_stream()

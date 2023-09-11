import pickle
import pyaudio
import librosa
import numpy as np
from time import sleep
from librosa import feature
from threading import Thread
import matplotlib.pyplot as plt


p = pyaudio.PyAudio()
# show devices
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    # print(d)

# select device for input and output
mic_device_index = 1

# WINDOW_SIZE = 2048
WINDOW_SIZE = 4096
CHANNELS = 1
RATE = 44100

FFT_FRAMES_IN_SPEC = 19

# global
global_blocks = np.zeros((FFT_FRAMES_IN_SPEC, WINDOW_SIZE))
fft_frame = np.array(WINDOW_SIZE // 2)
win = np.hamming(WINDOW_SIZE)
spec_img = np.zeros((WINDOW_SIZE // 2, FFT_FRAMES_IN_SPEC))

user_terminated = False


# ------------------------------------------------------------------------------------


def callback(in_data, frame_count, time_info, status):
    global global_blocks, fft_frame, win, spec_img
    # we don't have a file - just read in_data from mic
    # data_in is in the same format as the data retrieved from file,
    # in case the file is 16-bit
    # block_bytes_from_file = f.readframes(WINDOW_SIZE)
    numpy_block_from_bytes = np.frombuffer(in_data, dtype="int16")
    # begin with a zero buffer
    # begin with a zero buffer
    block_for_speakers = np.zeros(
        (numpy_block_from_bytes.size, CHANNELS), dtype="int16"
    )
    # pitch shift
    # block_for_speakers[:,0] = np.r_[ numpy_block_from_bytes[::2] , numpy_block_from_bytes[::2] ]
    # 0 is left, 1 is right speaker / channel
    block_for_speakers[:, 0] = numpy_block_from_bytes
    # for plotting
    # audio_data = np.fromstring(in_data, dtype=np.float32)
    if len(win) == len(numpy_block_from_bytes):
        frame_fft = np.fft.fft(win * numpy_block_from_bytes)
        p = np.abs(frame_fft) * 2 / np.sum(win)
        # translate in dB
        fft_frame = 20 * np.log10(p[: WINDOW_SIZE // 2] / 32678)
        spec_img = np.roll(spec_img, -1, axis=1)
        spec_img[:, -1] = fft_frame[::-1]
        global_blocks = np.roll(global_blocks, -1, axis=0)
        global_blocks[-1, :] = numpy_block_from_bytes
    return (block_for_speakers, pyaudio.paContinue)


def user_input_function():
    k = input('press "s" to terminate (then press "Enter"): ')
    print("pressed: ", k)
    if k == "s" or k == "S":
        global user_terminated
        user_terminated = True
        print("user_terminated 1: ", user_terminated)


# create output stream
output = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    input=True,
    input_device_index=mic_device_index,
    frames_per_buffer=WINDOW_SIZE,
    stream_callback=callback,
    start=False,
)

output.start_stream()

threaded_input = Thread(target=user_input_function)
threaded_input.start()

# load the pickle model
with open('best_clf.pkl', 'rb') as file:
    model = pickle.load(file)

# after starting, check when n empties (file ends) and stop

while output.is_active() and not user_terminated:

    plt.rcParams['figure.figsize'] = (10, 12)
    # plot Spectrogram
    plt.clf()
    plt.subplot(4, 1, 1)
    s = np.reshape(global_blocks, WINDOW_SIZE * FFT_FRAMES_IN_SPEC)
    plt.title("Spectrogram")

    # calculate Centroid
    cent = feature.spectral_centroid(
        y=s, sr=RATE, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE
    )

    plt.imshow(spec_img[WINDOW_SIZE // 4 :, :], aspect="auto")

    plt.subplot(4, 1, 2)
    times = librosa.times_like(cent)
    plt.plot(
        times,
        cent[0],
        label="Spectral centroid",
        color="r",
    )
    plt.title("Spectral centroid")

    # calculate MFCC
    mfcc = librosa.feature.mfcc(y=s, sr=RATE, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE)
    # print(mfcc)
    # plot MFCC
    plt.subplot(4, 1, 3)
    plt.imshow(mfcc, aspect="auto", origin="lower")
    plt.title("MFCC")

    # Predict
    predictions = model.predict(mfcc)
    # print(predictions)
    for prediction in predictions:
        # print(predictions)
        if prediction == 0:
            print('surprised')
        else:
            print('calm')

    # plot Signal

    plt.subplot(4, 1, 4)
    plt.title("Signal")
    plt.plot(s)
    plt.axis([0, s.size, -np.iinfo("int16").max, np.iinfo("int16").max])

    plt.subplots_adjust(hspace=0.5)
    plt.pause(0.01)


plt.show()

print("stopping audio")
output.stop_stream()

import tensorflow as tf
import numpy as np
import pyaudio
import turtle

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#####################################################################

def get_spectrogram(waveform):
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)

    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def preprocess_audiobuffer(waveform):
    # Normalize from [-32768, 32768] to [-1, 1]
    waveform = waveform / 32768
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    spectrogram = get_spectrogram(waveform)
    spectrogram = tf.expand_dims(spectrogram, 0)
    return spectrogram

#####################################################################

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAMES_PER_BUFFER = 3200
p = pyaudio.PyAudio()

def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER)

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    return np.frombuffer(b''.join(frames), dtype=np.int16)

def terminate():
    p.terminate()

#####################################################################

s = turtle.getscreen()
t = turtle.Turtle()

size = t.turtlesize()
increase = (2 * num for num in size)
t.turtlesize(*increase)

t.pensize(5)
t.shapesize()
t.pencolor("blue")

def go_right():
    current = t.heading()
    if current == 0:
        pass
    elif current == 90:
        t.right(90)
    elif current == 180:
        t.right(180)
    elif current == 270:
        t.left(90)
    else:
        raise ValueError('Not a right angle.')

def go_up():
    current = t.heading()
    if current == 0:
        t.left(90)
    elif current == 90:
        pass
    elif current == 180:
        t.right(90)
    elif current == 270:
        t.left(180)
    else:
        raise ValueError('Not a right angle.')

def go_left():
    current = t.heading()
    if current == 0:
        t.left(180)
    elif current == 90:
        t.left(90)
    elif current == 180:
        pass
    elif current == 270:
        t.right(90)
    else:
        raise ValueError('Not a right angle.')

def go_down():
    current = t.heading()
    if current == 0:
        t.right(90)
    elif current == 90:
        t.right(180)
    elif current == 180:
        t.left(90)
    elif current == 270:
        pass
    else:
        raise ValueError('Not a right angle.')

def move_turtle(command):
    if command == 'up':
        go_up()
    elif command =='down':
        go_down()
    elif command == 'left':
        go_left()
    elif command == 'right':
        go_right()
    elif command == 'go':
        t.forward(100)
    elif command == 'stop':
        print('Stopping the turtle.')

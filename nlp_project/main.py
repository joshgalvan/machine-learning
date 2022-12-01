import numpy as np
from tensorflow.keras import models
from helper_funcs import record_audio, terminate, preprocess_audiobuffer

# Implement commands from model.
commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

loaded_model = models.load_model('saved_model')

# Run helper commands to get live feed of microphone.
def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print(f"Predicted label: {command}")
    return command

# Grab live feed of mic, and operate on turtle.
if __name__ == "__main__":
    from helper_funcs import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break

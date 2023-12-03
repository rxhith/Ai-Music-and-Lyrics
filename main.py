import os
import glob
import numpy as np
from music21 import converter, note, chord
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

# Function to preprocess MIDI files and extract notes
def preprocess_midi(file_path):
    midi = converter.parse(file_path)

    notes = []

    # Extract notes and chords from MIDI file
    for element in midi.recurse():
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

# Function to prepare sequences for model training
def prepare_sequences(notes, sequence_length=100):
    unique_notes = sorted(set(notes))
    note_to_int = {note: i for i, note in enumerate(unique_notes)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(unique_notes))

    if len(network_output) > 0:
        network_output = to_categorical(network_output, num_classes=len(unique_notes))
    else:
        print("Warning: network_output is empty. Model training might fail.")

    return network_input, network_output, len(unique_notes)

# Directory containing MIDI files
midi_directory = r"C:\Users\user\Desktop\Rohith_Ai Music\midi file\maestro-v3.0.0\2004"

# Preprocess MIDI files and collect notes
notes_corpus = []
for file in glob.glob(os.path.join(midi_directory, "*.mid")):
    notes = preprocess_midi(file)
    notes_corpus.extend(notes)

# Prepare sequences for model training
sequence_length = 100  # Define the length of input sequences

if len(notes_corpus) > 0:  # Check if notes_corpus is not empty
    X, y, num_unique_notes = prepare_sequences(notes_corpus, sequence_length)

    if num_unique_notes > 0:  # Check if unique_notes is not empty
        # Define the model architecture
        model = Sequential()
        model.add(Embedding(input_dim=num_unique_notes, output_dim=100, input_length=sequence_length))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(num_unique_notes, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # Define a checkpoint to save the best model during training
        checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True, mode='min', verbose=1)

        # Train the model
        history = model.fit(X, y, epochs=50, batch_size=64, callbacks=[checkpoint])
    else:
        print("No unique notes found. Model training cannot proceed.")
else:
    print("No notes found in the MIDI files. Model training cannot proceed.")

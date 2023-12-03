import os
from music21 import converter, instrument, note, chord

def preprocess_midi(file_path):
    midi = converter.parse(file_path)

    notes = []
    instruments = []
    tempo_val = None
    time_signature = None
    key_signature = None

    # Extract notes and chords from MIDI file
    for element in midi.recurse():
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            chord_notes = '.'.join(str(n.pitch) for n in element.notes)
            notes.append(chord_notes)

        # Extract instruments
        if isinstance(element, instrument.Instrument):
            instruments.append(str(element.instrumentName))

        # Extract tempo, time signature, and key signature
        if isinstance(element, note.Note) and not tempo_val:  # Extract tempo from the first note
            tempo_val = element.quarterLength

        if isinstance(element, note.Note) and not time_signature:  # Extract time signature from the first note
            time_signature = element.getContextByClass('TimeSignature')

        if isinstance(element, note.Note) and not key_signature:  # Extract key signature from the first note
            key_signature = element.getContextByClass('KeySignature')

    return {
        'Notes': notes,
        'Instruments': instruments,
        'Tempo': tempo_val,
        'Time Signature': time_signature,
        'Key Signature': key_signature
    }

def preprocess_files_in_directory(directory, num_files_to_process=10):
    processed_files = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if processed_files >= num_files_to_process:
                break

            if file.endswith(".mid") or file.endswith(".midi"):
                file_path = os.path.join(root, file)
                data = preprocess_midi(file_path)
                # Process or display the extracted data as needed
                print(f"Processed file: {file_path}")
                for key, value in data.items():
                    print(f"{key}: {value}")
                print("---------------------------------------")

                processed_files += 1

        if processed_files >= num_files_to_process:
            break

# Directory containing MIDI files
midi_directory = r"C:\Users\user\Desktop\Rohith_Ai Music\midi file\maestro-v3.0.0\2004"

# Process and display notes from 10 MIDI files in the directory
preprocess_files_in_directory(midi_directory, num_files_to_process=5)

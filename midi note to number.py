#based on chart converts midi notes such as c#,d etc..to their corresponding midinumber
from midiutil import MIDIFile

def create_midi_file(notes, output_file='output.mid', tempo=120, repeat_count=3, note_duration=1):
    midi = MIDIFile(1)  # 1 track
    midi.addTempo(0, 0, tempo)

    for _ in range(repeat_count):
        time_step = 0  # in beats

        for note in notes:
            midi.addNote(0, 0, note_to_midi(note), time_step, note_duration, 100)  # channel, pitch, time, duration, volume
            time_step += note_duration

    with open(output_file, 'wb') as midi_file:
        midi.writeFile(midi_file)

def note_to_midi(note):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(note[-1])
    note_name = note[:-1]
    midi_number = (octave + 1) * 12 + note_names.index(note_name)
    return midi_number

if __name__ == "__main__":
    # Example notes (replace with your own)
    piano_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']

    output_midi_file = 'output2.mid'

    create_midi_file(piano_notes, output_file=output_midi_file, repeat_count=3, note_duration=1.5)

    print("MIDI file created:", output_midi_file)

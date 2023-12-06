#this code generates a midi file from midi numbers
#sample midi numbers are taken to gnerate midifiles
#better duration optimized
from midiutil import MIDIFile

def create_midi_file(notes, output_file='output.mid', tempo=120, repeat_count=3, note_duration=1):
    midi = MIDIFile(1)  # 1 track
    midi.addTempo(0, 0, tempo)

    for _ in range(repeat_count):
        time_step = 0  # in beats

        for note in notes:
            midi.addNote(0, 0, note, time_step, note_duration, 100)  # channel, pitch, time, duration, volume
            time_step += note_duration

    with open(output_file, 'wb') as midi_file:
        midi.writeFile(midi_file)

if __name__ == "__main__":
    # Example notes (replace with your own)
    piano_notes = [60, 62, 64, 65, 67, 69, 71, 72]

    output_midi_file = 'output4.mid'

    create_midi_file(piano_notes, output_file=output_midi_file, repeat_count=3, note_duration=1.5)

    print("MIDI file created:", output_midi_file)

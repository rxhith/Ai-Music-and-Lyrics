#this code plays the midi file that has been generated
#doesnt work in colab
import pygame
import time
import os

def play_midi(midi_file):
    os.environ['SDL_AUDIODRIVER'] = 'directsound'

    pygame.init()
    pygame.mixer.init()

    try:
        pygame.mixer.music.load(midi_file)
        print(f"Playing {midi_file}")
        pygame.mixer.music.play()

        # Wait for the music to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(1)

    except pygame.error as e:
        print(f"Error playing MIDI file: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    midi_file_path = 'output4.mid'  # Replace with the path to your MIDI file
    play_midi(midi_file_path)

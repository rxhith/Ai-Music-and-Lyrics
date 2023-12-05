import pygame
import time


def play_midi(midi_file):
    pygame.init()
    pygame.mixer.init()

    try:
        pygame.mixer.music.load(midi_file)
        print(f"Playing {midi_file}")
        pygame.mixer.music.play()

        # Allow time for the MIDI to play
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except pygame.error as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()


if __name__ == "__main__":

    midi_file_path = 'islamei.mid'

    play_midi(midi_file_path)

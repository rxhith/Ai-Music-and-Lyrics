from gtts import gTTS
from pydub import AudioSegment
import os

def text_to_playful_singing_audio(text, output_file='output.mp3', lang='en', speed=1, music_file='background_music.mp3'):
    # Generate playful singing audio
    styled_text = f"🎶 {text} 🎶"
    tts = gTTS(text=styled_text, lang=lang, slow=False)
    tts.speed = speed
    tts.save('temp_voice.mp3')

    # Load the generated voice and background music
    voice = AudioSegment.from_mp3('temp_voice.mp3')
    background_music = AudioSegment.from_mp3(music_file)

    # Adjust the length of the generated voice to match the background music
    voice = voice[:len(background_music)]

    # Combine the voice and background music
    combined_audio = voice.overlay(background_music)

    # Save the final audio
    combined_audio.export(output_file, format="mp3")

    # Clean up temporary files
    os.remove('temp_voice.mp3')

    return output_file

if __name__ == "__main__":
    # Ask the user to input their own lyrics
    user_lyrics = input("Enter your creative lyrics: ")

    # Specify the path to your background music file
    background_music_file = '/content/Ed Sheeran - Perfect (minus 24).mp3'

    # Generate playful singing audio with background music
    output_audio_file = text_to_playful_singing_audio(user_lyrics, music_file=background_music_file)

    # Provide a link to download the audio file
    print(f"Audio file generated. Click the link to download and play:\n{output_audio_file}")

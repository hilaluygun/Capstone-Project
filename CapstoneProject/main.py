import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import tempfile
import ffmpeg as ffm
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Streamlit app
st.title("Movie Transcription and Translation")

# File uploader
Movie_file = st.file_uploader("Upload a Movie file", type=['mp4', 'avi', 'mov'])

# Language selection
language = st.text_input("Enter the language for translation:")

def convert_video_to_audio(video_path):
    audio_path = video_path.rsplit('.', 1)[0] + '.mp3'
    if os.path.exists(audio_path):
        os.remove(audio_path)
    try:
        logging.info(f"Converting {video_path} to {audio_path}")
        stream = ffm.input(video_path)
        stream = ffm.output(stream, audio_path)
        ffm.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        logging.info("Conversion successful")
        return audio_path
    except ffm.Error as e:
        logging.error(f'FFmpeg error: {e.stderr.decode()}')
        return None
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        return None

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                logging.info(f"Temporary file saved: {tmp_file.name}")
                return tmp_file.name
        except Exception as e:
            logging.error(f"Error saving temporary file: {str(e)}")
            return None
    return None

# Process button
if st.button("Transcribe and Translate") and Movie_file is not None and language:
    with st.spinner("Processing audio..."):
        try:
            # Save uploaded file
            temp_Movie_path = save_uploaded_file(Movie_file)
            if temp_Movie_path is None:
                raise Exception("Failed to save uploaded file")

            temp_Audio_path = convert_video_to_audio(temp_Movie_path)
            if temp_Audio_path is None:
                raise Exception("Failed to convert video to audio")

            # Transcribe audio
            with open(temp_Audio_path, "rb") as Audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=Audio_file,
                    response_format="srt"
                )
            st.success("Audio transcribed successfully!")

            # Translate transcription
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",  # Updated to a more recent model
                messages=[
                    {"role": "system", "content": "You are a very helpful and talented translator who can translate all languages and srt files."},
                    {"role": "user", "content": f"Could you please translate the .srt text below to {language}? Do not add any comments of yours only the translation. "
                                                f"Please do not change the timestamps and structure of the file.\n<Transcription>{transcription}</Transcription>"}
                ]
            )
            translated_srt = response.choices[0].message.content
            st.success("Translation completed!")

            # Display translated subtitles
            st.subheader("Translated Subtitles")
            st.text_area("SRT Content", translated_srt, height=300)

            # Download button for translated SRT
            st.download_button(
                label="Download Translated SRT",
                data=translated_srt,
                file_name="translated_subtitles.srt",
                mime="text/plain"
            )

            # Clean up temporary files
            os.unlink(temp_Movie_path)
            os.unlink(temp_Audio_path)
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload an audio file (mp3, wav, or m4a format).
2. Enter the desired language for translation.
3. Click 'Transcribe and Translate' to process the audio.
4. View the translated subtitles and download the SRT file.
""")



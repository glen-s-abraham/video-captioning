import subprocess
import os
import logging
import openai
import tempfile
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)

# OpenAI API Key for transcription
openai.api_key = 'api-key'

def extract_audio_from_video(video_file_path, output_audio_path):
    """
    Extracts audio from a video file using FFmpeg.

    Parameters
    ----------
    video_file_path : str
        Path to the source video file.
    output_audio_path : str
        Path where the extracted audio will be saved.
    
    Returns
    -------
    None
    
    Raises
    ------
    OSError
        If FFmpeg fails to extract the audio.
    """
    logging.info(f"Extracting audio from {video_file_path} to {output_audio_path}")
    
    command = ["ffmpeg", "-i", video_file_path, "-q:a", "0", "-map", "a", output_audio_path]
    
    try:
        subprocess.run(command, check=True)
        logging.info(f"Audio extracted to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to extract audio from video")
        raise OSError("Failed to extract audio from video") from e

def transcribe_and_save_srt(audio_file_path, input_video_name):
    """
    Transcribe the given audio file and save the transcript in SRT format.

    Parameters
    ----------
    audio_file_path : str
        Path to the audio file.
    input_video_name : str
        Name of the input video (without extension) to tie the SRT file name to the video.

    Returns
    -------
    str
        Path to the saved SRT file.
    
    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    """
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file {audio_file_path} does not exist")

    logging.info(f"Transcribing audio file {audio_file_path}")
    
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",  # hypothetical model name
            file=audio_file,
            response_format='srt'  # Get SRT formatted response
        )
    
    srt_path = tempfile.mktemp(suffix=".srt")
    
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    logging.info(f"SRT file saved as {srt_path}")
    return srt_path

def embed_subtitle_in_video(video_file_path, subtitle_file_path, output_file_path):
    """
    Embed the subtitle file into the video using FFmpeg.

    Parameters
    ----------
    video_file_path : str
        Path to the source video file.
    subtitle_file_path : str
        Path to the subtitle (SRT) file.
    output_file_path : str
        Path where the output video with subtitles will be saved.
    
    Returns
    -------
    None
    
    Raises
    ------
    OSError
        If FFmpeg fails to embed the subtitles.
    """
    logging.info(f"Embedding subtitles from {subtitle_file_path} into video {video_file_path}")

    command = [
        "ffmpeg",
        "-i", video_file_path,
        "-vf", f"subtitles={subtitle_file_path}",
        output_file_path
    ]
    
    try:
        subprocess.run(command, check=True)
        logging.info(f"Video with subtitles saved as {output_file_path}")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to embed subtitles in video")
        raise OSError("Failed to embed subtitles in video") from e

def main():
    st.title("Video Captioning App")
    
    uploaded_videos = st.file_uploader("Upload video files", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True)
    
    if uploaded_videos and st.button("Generate Captions"):
        # Create temp directory if it doesn't exist
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each uploaded video file
            for uploaded_video in uploaded_videos:
                video_file_path = os.path.join(temp_dir, uploaded_video.name)
                output_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
                output_video_path = os.path.join(temp_dir, f"output_{uploaded_video.name}")
                
                # Save the uploaded video to temp directory
                with open(video_file_path, "wb") as f:
                    f.write(uploaded_video.read())
                
                try:
                    # Step 1: Extract audio from video
                    extract_audio_from_video(video_file_path, output_audio_path)

                    # Step 2: Transcribe the extracted audio and save as SRT
                    subtitle_file_path = transcribe_and_save_srt(output_audio_path, uploaded_video.name)

                    # Step 3: Embed subtitles into the original video
                    embed_subtitle_in_video(video_file_path, subtitle_file_path, output_video_path)
                
                    # Provide download link for the output video
                    with open(output_video_path, "rb") as f:
                        st.download_button(label=f"Download {uploaded_video.name} with Subtitles",
                                           data=f,
                                           file_name=f"output_{uploaded_video.name}",
                                           mime="video/mp4")
                    
                    # Display the processed video
                    st.video(output_video_path)

                except FileNotFoundError as e:
                    st.error(f"Error processing {uploaded_video.name}: {e}")
                except OSError as e:
                    st.error(f"Error processing {uploaded_video.name}: {e}")

                # Cleanup temporary files
                try:
                    os.remove(video_file_path)
                    os.remove(output_audio_path)
                    os.remove(subtitle_file_path)
                except OSError as cleanup_error:
                    logging.warning(f"Failed to delete temporary file: {cleanup_error}")

if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
from urllib.parse import urlparse

def _is_youtube_url(url : str) -> bool:
    """Check if the given URL is a YouTube URL."""
    parsed_url = urlparse(url)
    return parsed_url.netloc.endswith('youtube.com') or parsed_url.netloc.endswith('youtu.be')

def _extract_audio_from_youtube(
        url : str, 
        filename : str = None, 
        output_path : str ='data/audio'
    ) -> None:
    """
    Downloads audio from a YouTube URL.
    Uses yt-dlp to download the audio from the given YouTube URL.
    The audio is saved in the specified output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if filename:
        output_template = os.path.join(output_path, f'{filename}.%(ext)s')
    else:
        output_template = os.path.join(output_path, '%(title)s.%(ext)s')

    command = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'mp3',
        '-o', output_template,
        url
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Audio downloaded from {url} and saved in {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio from YouTube: {e}")
    except FileNotFoundError:
        print("Error: yt-dlp is not installed or not in your PATH.")
        print("Please install it using: pip install yt-dlp")


def _extract_audio_from_mp4(
        file_path : str, 
        filename : str = None, 
        output_path : str ='data/audio'
    ):
    """
    Extracts audio from an MP4 file.
    Uses moviepy to extract the audio from the given MP4 file.
    The audio is saved as an MP3 file in the specified output path.
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except ImportError as e:
        print(f"Error: moviepy could not be imported: {e}")
        print("Please install it using: pip install moviepy")
        return

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_clip = VideoFileClip(file_path)
    audio_clip = video_clip.audio
    
    if audio_clip is None:
        print(f"Error: No audio found in the video file {file_path}")
        video_clip.close()
        return

    if not filename:
        base_name = os.path.basename(file_path)
        filename = os.path.splitext(base_name)[0]

    output_filename = os.path.join(output_path, f"{filename}.mp3")
    
    audio_clip.write_audiofile(output_filename)
    
    audio_clip.close()
    video_clip.close()
    
    print(f"Audio extracted from {file_path} and saved as {output_filename}")


def extract_audio(
        source : str, 
        filename : str = None, 
        output_path : str = 'data/audio'
    ):
    """
    Extracts audio from a YouTube URL or an MP4 file.
    This is the main exportable function.
    """
    if _is_youtube_url(source):
        _extract_audio_from_youtube(source, filename, output_path)
    elif os.path.isfile(source) and source.lower().endswith('.mp4'):
        _extract_audio_from_mp4(source, filename, output_path)
    else:
        print(f"Error: Unsupported source type or file not found: {source}")
        print("Please provide a valid YouTube URL or a path to an .mp4 file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio from a YouTube link or an MP4 file.')
    parser.add_argument('source', help='A YouTube URL or a path to an MP4 file.')
    parser.add_argument('-o', '--output', default='data/audio', help='The output directory for the audio file.')
    parser.add_argument('-f', '--filename', help='The filename for the output audio file (without extension).')
    
    args = parser.parse_args()
    
    extract_audio(args.source, filename=args.filename, output_path=args.output)

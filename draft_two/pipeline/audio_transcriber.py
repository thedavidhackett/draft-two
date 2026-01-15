import modal
import os
import argparse
from dotenv import load_dotenv

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

load_dotenv()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "git+https://github.com/m-bain/whisperX.git",
        "torch",
        "torchaudio",
        "python-dotenv",
        "omegaconf"
    )
)

app = modal.App("whisperx-transcription-service")

@app.cls(
    image=image,
    gpu="A10G",
)
class WhisperXWorker:
    @modal.method()
    def process_audio(self, audio_bytes: bytes):
        import whisperx
        import tempfile
        import os
        
        device = "cuda"
        batch_size = 16 
        compute_type = "float16"

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        try:
            # 1. Transcribe
            model = whisperx.load_model("large-v3", device, compute_type=compute_type)
            audio = whisperx.load_audio(temp_path)
            result = model.transcribe(audio, batch_size=batch_size)
            return result["segments"]
        
        finally:
            os.remove(temp_path)

def main(source: str, output: str = 'data/text_files', filename: str = None, num_repeats: int = 5):
    """
    Local entrypoint for the audio transcription script.
    Transcribes an audio file N times and saves each transcription to a separate file
    in a dedicated folder.
    """
    if not os.path.isfile(source) or not source.lower().endswith('.mp3'):
        print(f"Error: Unsupported file type or file not found: {source}")
        print("Please provide a path to an .mp3 file.")
        return

    if not filename:
        base_name = os.path.basename(source)
        filename = os.path.splitext(base_name)[0]

    # Create a dedicated output folder for this batch of transcriptions
    output_folder_path = os.path.join(output, filename)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    print(f"Reading audio file: {source}")
    with open(source, "rb") as f:
        audio_bytes = f.read()

    print(f"Submitting {num_repeats} transcription jobs to Modal...")
    worker = WhisperXWorker()
    
    # Use .map to run multiple transcriptions in parallel
    all_segments = worker.process_audio.map([audio_bytes] * num_repeats)

    print(f"Transcription complete. Saving {num_repeats} files to {output_folder_path}")
    for i, segments in enumerate(all_segments):
        output_filename = os.path.join(output_folder_path, f"{filename}-{i+1}.txt")
        with open(output_filename, "w") as f:
            full_text = " ".join(segment['text'].strip() for segment in segments)
            f.write(full_text)
    
    print(f"All {num_repeats} transcriptions saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe an audio file N times using WhisperX on Modal.')
    parser.add_argument('source', help='A path to an MP3 file.')
    parser.add_argument('-o', '--output', default='data/text_files', help='The parent output directory for the transcription folder.')
    parser.add_argument('-f', '--filename', help='The base filename for the output folder and files.')
    parser.add_argument('-n', '--num_repeats', type=int, default=5, help='Number of times to transcribe the audio.')

    args = parser.parse_args()
    with app.run():
        main(args.source, output=args.output, filename=args.filename, num_repeats=args.num_repeats)

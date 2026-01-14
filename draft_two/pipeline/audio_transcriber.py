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

def main(source: str, output: str = 'data/text_files', filename: str = None):
    """
    Local entrypoint for the audio transcription script.
    """
    if not os.path.isfile(source) or not source.lower().endswith('.mp3'):
        print(f"Error: Unsupported file type or file not found: {source}")
        print("Please provide a path to an .mp3 file.")
        return

    if not os.path.exists(output):
        os.makedirs(output)

    print(f"Reading audio file: {source}")
    with open(source, "rb") as f:
        audio_bytes = f.read()

    print("Submitting transcription job to Modal...")
    worker = WhisperXWorker()
    segments = worker.process_audio.remote(audio_bytes)

    if not filename:
        base_name = os.path.basename(source)
        filename = os.path.splitext(base_name)[0]

    output_filename = os.path.join(output, f"{filename}.txt")

    print(f"Transcription complete. Saving to {output_filename}")
    with open(output_filename, "w") as f:
        # Join all segment texts with a space to form a single block of text.
        full_text = " ".join(segment['text'].strip() for segment in segments)
        f.write(full_text)
    
    print("Transcription saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe an audio file using WhisperX on Modal.')
    parser.add_argument('source', help='A path to an MP3 file.')
    parser.add_argument('-o', '--output', default='data/text_files', help='The output directory for the text file.')
    parser.add_argument('-f', '--filename', help='The filename for the output text file (without extension).')

    args = parser.parse_args()
    with app.run():
        main(args.source, output=args.output, filename=args.filename)
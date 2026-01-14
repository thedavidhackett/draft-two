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
        "whisperx",
        "torch",
        "torchaudio",
        "pyannote.audio",
        "python-dotenv",
        "omegaconf"
    )
)

app = modal.App("whisperx-diarization-service")

@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HUGGING_FACE_HUB_TOKEN", "")})]
)
class WhisperXWorker:
    @modal.method()
    def process_audio(self, audio_bytes: bytes, num_speakers: int = None):
        import whisperx
        import tempfile
        import os
        # import torch
        # import typing
        # from omegaconf import ListConfig, DictConfig
        # from omegaconf.base import ContainerMetadata, Node

        # # Allowlist the ListConfig class for torch.load
        # torch.serialization.add_safe_globals([
        #             ListConfig, 
        #             DictConfig, 
        #             ContainerMetadata, 
        #             Node,
        #             typing.Any
        #         ])
        
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

            # 2. Align (Word-level timestamps)
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, device, return_char_alignments=False
            )

            # 3. Diarize (Speaker Identification)
            # Make sure we have a token for pyannote
            try:
                hf_token = os.environ["HF_TOKEN"]
                if not hf_token:
                    raise KeyError
            except KeyError:
                raise RuntimeError("HF_TOKEN environment variable not set. Please set it in your Modal secret or .env file.")

            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
            
            # 4. Final Assignment
            result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
            return result_with_speakers["segments"]
        
        finally:
            os.remove(temp_path)

def main(source: str, output: str = 'data/text_files', filename: str = None, num_speakers: int = None):
    """
    Local entrypoint for the audio transcription and diarization script.
    """
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") == "YOUR_HUGGING_FACE_TOKEN_HERE":
        print("Error: HUGGING_FACE_HUB_TOKEN not set in .env file.")
        print("Please create a .env file and add your Hugging Face Hub token.")
        return

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
    segments = worker.process_audio.remote(audio_bytes, num_speakers)

    if not filename:
        base_name = os.path.basename(source)
        filename = os.path.splitext(base_name)[0]

    output_filename = os.path.join(output, f"{filename}.txt")

    print(f"Transcription complete. Saving to {output_filename}")
    with open(output_filename, "w") as f:
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment['text'].strip()
            # To get a timestamp, we can take the start of the segment
            start_time = segment['start']
            # format time as H:M:S.ms
            start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:06.3f}"
            f.write(f"[{start_time_str}] {speaker}: {text}\n")
    
    print("Transcription saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe and diarize an audio file using WhisperX on Modal.')
    parser.add_argument('source', help='A path to an MP3 file.')
    parser.add_argument('-o', '--output', default='data/text_files', help='The output directory for the text file.')
    parser.add_argument('-f', '--filename', help='The filename for the output text file (without extension).')
    parser.add_argument('-n', '--num_speakers', type=int, help='The number of speakers in the audio file.')

    args = parser.parse_args()
    with app.run():
        main(args.source, output=args.output, filename=args.filename, num_speakers=args.num_speakers)

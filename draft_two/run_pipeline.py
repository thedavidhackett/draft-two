import argparse
import os
import subprocess
import sys

def run_command(command):
    """Runs a command and prints its output in real-time."""
    print(f"\n{'='*20}")
    print(f"Running command: {' '.join(command)}")
    print(f"{ '='*20}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        rc = process.poll()
        if rc != 0:
            print(f"Error: Command failed with exit code {rc}")
            sys.exit(1) # Exit the pipeline if a step fails
            
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():

    parser = argparse.ArgumentParser(description="Orchestrates the full data processing pipeline.")

    parser.add_argument("source", help="The source YouTube URL or local .mp4 file path.")

    parser.add_argument("filename", help="The base filename (without extension) to use for all generated files.")

    parser.add_argument("-i", "--instructions_file", default="data/raw/instructions.txt", help="Path to the instructions file for the batch processor.")

    parser.add_argument("-r", "--num_repeats", type=int, default=5, help="Number of times to repeat the request in the batch processor.")



    args = parser.parse_args()



    # --- Define Paths ---

    # These paths are based on the default output paths in the individual scripts.

    audio_output_dir = "data/audio"

    transcribed_text_output_dir = "data/text_files"

    metadata_output_dir = "data/metadata"

    reports_output_dir = "data/ai_police_reports"

    facts_output_dir = "data/atomic_facts"

    

    # Ensure the Python executable from the current environment is used

    python_executable = sys.executable



    # Construct the full file paths that will be generated and used

    audio_file_path = os.path.join(audio_output_dir, f"{args.filename}.mp3")

    text_file_path = os.path.join(transcribed_text_output_dir, f"{args.filename}.txt")

    metadata_file_path = os.path.join(metadata_output_dir, f"{args.filename}_metadata.txt")

    report_folder_path = os.path.join(reports_output_dir, args.filename)



    # --- Pipeline Steps ---

    

    # 1. Audio Extractor

    print("--- Step 1: Extracting Audio ---")

    audio_extractor_cmd = [

        python_executable, "draft_two/pipeline/audio_extractor.py",

        args.source,

        "-f", args.filename,

        "-o", audio_output_dir

    ]

    run_command(audio_extractor_cmd)



    # 2. Audio Transcriber

    print("\n--- Step 2: Transcribing Audio ---")

    audio_transcriber_cmd = [

        python_executable, "draft_two/pipeline/audio_transcriber.py",

        audio_file_path,

        "-f", args.filename,

        "-o", transcribed_text_output_dir

    ]

    run_command(audio_transcriber_cmd)



    # 3. Metadata Creator (Interactive)

    print("\n--- Step 3: Creating Metadata (Interactive) ---")

    metadata_creator_cmd = [

        python_executable, "draft_two/pipeline/metadata_creator.py",

        args.filename,

        "-o", metadata_output_dir

    ]

    run_command(metadata_creator_cmd)



    # 4. Batch Processor

    print("\n--- Step 4: Processing Reports in Batch ---")

    batch_processor_cmd = [

        python_executable, "draft_two/pipeline/batch_processor.py",

        text_file_path,

        "-i", args.instructions_file,

        "-m", metadata_file_path,

        "-o", reports_output_dir,

        "-n", str(args.num_repeats)

    ]

    run_command(batch_processor_cmd)



    # 5. Fact Extractor

    print("\n--- Step 5: Extracting Atomic Facts ---")

    fact_extractor_cmd = [

        python_executable, "draft_two/pipeline/fact_extractor.py",

        report_folder_path,

        "-o", facts_output_dir

    ]

    run_command(fact_extractor_cmd)



    print(f"\n{'='*20}")

    print("Pipeline completed successfully!")

    print(f"{'='*20}")

if __name__ == "__main__":
    main()

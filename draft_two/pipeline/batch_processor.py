import os
import argparse
import json
import time
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

def process_batch(
        input_folder_path: str,
        instructions_file_path: str,
        output_path: str = 'data/ai_police_reports',
        metadata_file_path: str = None
    ):
    """
    Creates an OpenAI batch job to process a folder of text files based on instructions,
    waits for completion, and saves the results.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        print("Error: OPENAI_API_KEY not set in .env file.")
        print("Please create a .env file and add your OpenAI API key.")
        return

    client = OpenAI(api_key=api_key)

    # The base_filename for the output folder is derived from the input folder's name
    base_filename = os.path.basename(input_folder_path)

    # 1. Read shared input files and identify transcription files
    try:
        with open(instructions_file_path, 'r') as f:
            system_prompt = f.read()

        # Handle metadata
        if metadata_file_path is None:
            # Construct default metadata path based on the folder name
            metadata_file_path = f"data/metadata/{base_filename}_metadata.txt"
            print(f"No metadata file provided. Using default path: {metadata_file_path}")

        try:
            with open(metadata_file_path, 'r') as f:
                metadata_content = f.read()
        except FileNotFoundError:
            metadata_content = ""
            print(f"Warning: Metadata file not found at {metadata_file_path}. Proceeding without metadata.")
        
        transcription_files = [f for f in os.listdir(input_folder_path) if f.endswith('.txt')]
        if not transcription_files:
            print(f"Error: No .txt files found in {input_folder_path}")
            return

    except FileNotFoundError as e:
        print(f"Error: Input file/folder not found - {e}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # 2. Prepare batch data by looping through transcription files
    batch_requests = []
    for txt_filename in transcription_files:
        try:
            with open(os.path.join(input_folder_path, txt_filename), 'r') as f:
                text_content = f.read()
        except IOError as e:
            print(f"Warning: Could not read file {txt_filename}. Skipping. Error: {e}")
            continue
            
        user_prompt = (
            "Based on the following metadata and transcript write a police report\n"
            f"Metadata:\n{metadata_content}\n"
            f"Transcript:\n{text_content}"
        )
        
        custom_id = os.path.splitext(txt_filename)[0]

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0
            }
        }
        batch_requests.append(request)

    if not batch_requests:
        print("Error: No valid batch requests could be created.")
        return

    # 3. Upload batch file
    with tempfile.NamedTemporaryFile(mode='w+b', suffix=".jsonl") as temp_batch_file:
        for request in batch_requests:
            temp_batch_file.write((json.dumps(request) + "\n").encode('utf-8'))

        temp_batch_file.seek(0)

        print(f"Batch data for {len(batch_requests)} requests prepared in a temporary file.")

        print("Uploading batch file to OpenAI...")
        batch_input_file = client.files.create(
            file=temp_batch_file.read(),
            purpose="batch"
        )
        print(f"File uploaded with ID: {batch_input_file.id}")


    # 4. Create batch job
    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Batch job created with ID: {batch.id}")

    # 5. Wait for completion
    print("Waiting for batch job to complete... (this may take a while)")
    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"Current batch status: {batch.status}")
        if batch.status in ["completed", "failed", "cancelled"]:
            break
        time.sleep(30) # Poll every 30 seconds

    if batch.status != "completed":
        print(f"Batch job did not complete successfully. Final status: {batch.status}")
        if batch.errors:
            print("Errors:", batch.errors)
        return

    print("Batch job completed successfully.")

    # 6. Retrieve and save results
    output_file_id = batch.output_file_id
    if not output_file_id:
        print("Error: No output file ID found in completed batch.")
        return

    print(f"Retrieving results from file ID: {output_file_id}")
    result_content_response = client.files.content(output_file_id)
    result_content = result_content_response.read().decode('utf-8')

    # Create a dedicated folder for the results, named after the input folder
    output_folder_path = os.path.join(output_path, base_filename)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    print(f"Saving results to folder: {output_folder_path}")
    try:
        # The result content is a JSONL file.
        lines = result_content.strip().split('\n')
        for line in lines:
            json_line = json.loads(line)
            custom_id = json_line.get('custom_id', 'unknown_request')
            content = json_line['response']['body']['choices'][0]['message']['content']
            
            # Create a separate file for each result
            output_filename = os.path.join(output_folder_path, f"{custom_id}.txt")
            with open(output_filename, 'w') as f:
                f.write(content)

        print("All results saved in separate files.")

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing result file: {e}")
        print("Saving raw result content for debugging...")
        raw_error_path = os.path.join(output_folder_path, "raw_error_output.txt")
        with open(raw_error_path, 'w') as f:
            f.write(result_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit a folder of text files and instructions to OpenAI for batch processing.')
    parser.add_argument('input_folder', help='Path to the folder of text files to be processed.')
    parser.add_argument('-i', '--instructions_file', default='data/raw/instructions.txt', help='Path to the file containing instructions for the model.')
    parser.add_argument('-o', '--output', default='data/ai_police_reports', help='The output directory for the processed file.')
    parser.add_argument('-m', '--metadata_file', help='Path to the metadata file.')

    args = parser.parse_args()
    
    process_batch(
        args.input_folder, 
        args.instructions_file, 
        args.output, 
        args.metadata_file
    )
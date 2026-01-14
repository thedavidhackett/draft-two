import os
import argparse
import json
import time
import tempfile
import csv
from openai import OpenAI
from dotenv import load_dotenv

def extract_facts_in_batch(
        input_folder_path: str,
        output_path: str = 'data/atomic_facts',
    ):
    """
    Creates an OpenAI batch job to extract atomic facts from each text file in a folder,
    waits for completion, and saves the results to CSV files.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        print("Error: OPENAI_API_KEY not set in .env file.")
        print("Please create a .env file and add your OpenAI API key.")
        return

    client = OpenAI(api_key=api_key)

    # 1. Read input files from the folder
    try:
        files_to_process = [f for f in os.listdir(input_folder_path) if f.endswith('.txt')]
        if not files_to_process:
            print(f"No .txt files found in {input_folder_path}")
            return
    except FileNotFoundError:
        print(f"Error: Input folder not found - {input_folder_path}")
        return

    # 2. Prepare batch data
    batch_requests = []
    for filename in files_to_process:
        file_path = os.path.join(input_folder_path, filename)
        try:
            with open(file_path, 'r') as f:
                text_content = f.read()
        except FileNotFoundError:
            print(f"Warning: Could not read file {file_path}. Skipping.")
            continue

        base_filename = os.path.splitext(filename)[0]
        user_prompt = f"Please breakdown the following report into independent facts: {text_content}"

        request = {
            "custom_id": base_filename,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5.2",
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
            }
        }
        batch_requests.append(request)

    if not batch_requests:
        print("No valid requests to process.")
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

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"Saving results to folder: {output_path}")
    try:
        # The result content is a JSONL file.
        lines = result_content.strip().split('\n')
        for line in lines:
            json_line = json.loads(line)
            custom_id = json_line.get('custom_id', 'unknown_request')
            content = json_line['response']['body']['choices'][0]['message']['content']
            
            # The content should be a list of facts, separated by newlines
            facts = [fact.strip() for fact in content.strip().split('\n') if fact.strip()]
            
            # Create a separate CSV file for each result
            output_filename = os.path.join(output_path, f"{custom_id}.csv")
            with open(output_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Fact"]) # Header
                for fact in facts:
                    writer.writerow([fact])

        print("All results saved in separate CSV files.")

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing result file: {e}")
        print("Saving raw result content for debugging...")
        input_folder_name = os.path.basename(input_folder_path)
        raw_error_path = os.path.join(output_path, f"raw_error_output_{input_folder_name}.txt")
        with open(raw_error_path, 'w') as f:
            f.write(result_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract atomic facts from text files in a folder using OpenAI batch processing.')
    parser.add_argument('input_folder', help='Path to the folder containing text files to be processed.')
    parser.add_argument('-o', '--output', default='data/atomic_facts', help='The output directory for the processed CSV files.')

    args = parser.parse_args()
    
    extract_facts_in_batch(
        args.input_folder, 
        args.output, 
    )

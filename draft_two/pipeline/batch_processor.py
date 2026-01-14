import os
import argparse
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

def process_batch(text_file_path: str, instructions_file_path: str, output_path: str = 'data/ai_police_reports'):
    """
    Creates an OpenAI batch job to process a text file based on instructions,
    waits for completion, and saves the result.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        print("Error: OPENAI_API_KEY not set in .env file.")
        print("Please create a .env file and add your OpenAI API key.")
        return

    client = OpenAI(api_key=api_key)

    # 1. Read input files
    try:
        with open(instructions_file_path, 'r') as f:
            system_prompt = f.read()
        with open(text_file_path, 'r') as f:
            user_prompt = f.read()
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        return

    # 2. Prepare batch data
    batch_requests = [
        {
            "custom_id": "request-1",
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
    ]

    batch_file_path = "temp_batch_input.jsonl"
    with open(batch_file_path, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    print(f"Batch data prepared at {batch_file_path}")

    # 3. Upload batch file
    try:
        print("Uploading batch file to OpenAI...")
        batch_input_file = client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )
        print(f"File uploaded with ID: {batch_input_file.id}")
    finally:
        os.remove(batch_file_path) # Clean up temp file

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

    # Determine output filename
    base_name = os.path.basename(text_file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_filename = os.path.join(output_path, f"{file_name_without_ext}_processed.txt")

    print(f"Saving results to {output_filename}")
    try:
        # The result content is a JSONL file.
        lines = result_content.strip().split('\n')
        final_text = ""
        for line in lines:
            json_line = json.loads(line)
            # Extract the content from the assistant's message
            content = json_line['response']['body']['choices'][0]['message']['content']
            final_text += content + "\n"

        with open(output_filename, 'w') as f:
            f.write(final_text.strip())
        
        print("Results saved.")

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing result file: {e}")
        print("Saving raw result content for debugging...")
        with open(output_filename + ".raw_error", 'w') as f:
            f.write(result_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit a text file and instructions to OpenAI for batch processing.')
    parser.add_argument('text_file', help='Path to the text file to be processed.')
    parser.add_argument('instructions_file', help='Path to the file containing instructions for the model.')
    parser.add_argument('-o', '--output', default='data/ai_police_reports', help='The output directory for the processed file.')
    
    args = parser.parse_args()
    
    process_batch(args.text_file, args.instructions_file, args.output)

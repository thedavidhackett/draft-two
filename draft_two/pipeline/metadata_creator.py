import os
import argparse
import sys
from datetime import datetime

def get_validated_input(prompt, validation_func, error_message):
    """Generic function to get validated user input."""
    while True:
        print(prompt, end='', flush=True)
        user_input = sys.stdin.readline().strip()
        if validation_func(user_input):
            return user_input
        else:
            print(error_message, flush=True)

def get_choice_from_options(prompt, options):
    """Gets a user's choice from a list of options."""
    print(prompt, flush=True)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}", flush=True)

    def is_valid_choice(user_input):
        return user_input.isdigit() and 1 <= int(user_input) <= len(options)

    choice_index = int(get_validated_input(
        "Enter the number of your choice: ",
        is_valid_choice,
        f"Invalid input. Please enter a number between 1 and {len(options)}."
    )) - 1
    return options[choice_index]

def get_yes_no_input(prompt):
    """Gets a 'yes' or 'no' answer from the user."""
    def is_yes_no(user_input):
        return user_input.lower() in ['yes', 'no', 'y', 'n']

    response = get_validated_input(
        prompt + " (yes/no): ",
        is_yes_no,
        "Invalid input. Please enter 'yes' or 'no'."
    )
    return response.lower().startswith('y')

def main(filename, output="data/metadata"):
    """Main function to gather metadata and write to a file."""
    print("Please provide the following information for the incident report.", flush=True)

    # Incident Type
    incident_types = [
        "Domestic Dispute",
        "Drug Related",
        "Fraud and Financial",
        "Impaired Driving",
        "Informational Report",
        "Missing Person",
        "Property Crime",
        "Public Disorder",
        "Sexual Offense",
        "Traffic Incident",
        "Violent Crime",
        "Other"
    ]

    charge_severities = ["Misdemeanor", "Felony", "Infraction", "No Charges"]

    print(f"Enter the incident date (YYYY-MM-DD, default: today): ", end='', flush=True)
    incident_date = sys.stdin.readline().strip() or datetime.now().strftime('%Y-%m-%d')
    
    incident_type = get_choice_from_options("\nSelect the incident type:", incident_types)
    charge_severity = get_choice_from_options("\nSelect the charge severity:", charge_severities)
    arrest_made = get_yes_no_input("\nWas an arrest made?")

    metadata = {
        "Incident Date": incident_date,
        "Incident Type": incident_type,
        "Charge Severity": charge_severity,
        "Arrest Made": "Yes" if arrest_made else "No",
    }

    if not filename:
        print("Filename cannot be empty. Exiting.", flush=True)
        return

    if not os.path.exists(output):
        os.makedirs(output)    
        
    output_path = os.path.join(output, f'{filename}_metadata.txt')

    try:
        with open(output_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"\nMetadata successfully saved to {output_path}", flush=True)
    except IOError as e:
        print(f"Error writing to file: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create metadata details for an incident report.')
    parser.add_argument('filename', help='Name of the output file (without extension), will add _metadata.txt to the end.')
    parser.add_argument('-o', '--output', default='data/metadata', help='The output directory for the processed file.')
    args = parser.parse_args()
    
    main(args.filename, args.output)

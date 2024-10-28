import os

def concatenate_inst_files(output_file):
    """
    Concatenates the text from all .inst files in the current directory into a single file.
    For each file, it includes the filename and its contents in the specified format.
    """
    # Get a list of all .inst files in the current directory
    inst_files = [f for f in os.listdir('./data') if f.endswith('.inst')]
    # Optionally, sort the list of files
    inst_files.sort()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in inst_files:
            # Write the filename
            outfile.write(f"file {filename}:\n\n")
            # Read the contents of the .inst file
            with open("./data/"+filename, 'r', encoding='utf-8') as infile:
                contents = infile.read()
                outfile.write(contents)
                outfile.write("\n\n")  # Add spacing between files
    print(f"Concatenated {len(inst_files)} files into {output_file}.")

if __name__ == "__main__":
    output_file = 'concatenated_inst_files.txt'
    concatenate_inst_files(output_file)

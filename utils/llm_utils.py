def read_prompt_from_file(filepath) -> str:
    # Read the base prompt from the text file
    try:
        with open(filepath, "r") as file:
            return file.read().strip()  # Remove any leading/trailing whitespace
    except FileNotFoundError:
        raise FileNotFoundError("File was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the base prompt: {e}")

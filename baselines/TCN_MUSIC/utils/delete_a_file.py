"""Remove a single file after an optional legacy confirmation prompt."""

import os


def delete_a_file(file_path,
                  deletion_message=False):
    try:
        os.remove(file_path)
        if deletion_message:
            print(f"File '{file_path}' has been deleted successfully.")
        return True
    except FileNotFoundError:
        if deletion_message:
            print(f"File '{file_path}' not found.")
    except Exception as e:
        if deletion_message:
            print(f"An error occurred while deleting the file: {str(e)}")

    return False

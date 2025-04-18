def find_max_number(filename):
    """
    Read all numeric values from specified file and return the maximum
    :param filename: Path to the file containing numeric values (one per line)
    :return: Maximum numeric value found, or None if no valid numbers or errors occur
    """
    max_value = float('-inf')  # Initialize with negative infinity for comparison baseline

    try:
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):  # Start line numbering at 1
                line = line.strip()
                if not line:
                    continue  # Skip empty lines to avoid conversion attempts

                try:
                    number = float(line)
                    if number > max_value:
                        max_value = number  # Update current maximum
                except ValueError:
                    print(f"Warning: Line {line_num} contains invalid data: '{line}'")
                    continue

        if max_value == float('-inf'):
            print("Warning: No valid numeric values found in file")
            return None

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except Exception as e:  # Catch-all for unexpected I/O errors
        print(f"Unexpected error reading file: {str(e)}")
        return None

    return max_value


# Example usage with Windows/Linux compatible path format
if __name__ == "__main__":
    filename = "F:/unet-pytorch-main/logs/loss_2025_mobilenet+sgd/epoch_f1score_mobilenet+sgd.txt"
    result = find_max_number(filename)
    if result is not None:
        print(f"Maximum value found: {result}")
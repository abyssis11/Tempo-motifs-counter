import pandas as pd

def getEdges(file_path):
    """
    Reads a CSV file, extracts and transforms 'node_1', 'node_2', and 'start_of_interaction' columns,
    and returns a list of tuples structured as [((node_1, node_2), start_of_interaction), ...].

    :param file_path: Path to the CSV file
    :return: List of tuples with the structure [((node_1, node_2), start_of_interaction), ...]
    """
    def transform_node_value(node_val):
        # Remove 'fly' and convert to integer
        return int(node_val.replace('fly', ''))

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract and transform 'node_1' and 'node_2'
    df['node_1'] = df['node_1'].apply(transform_node_value)
    df['node_2'] = df['node_2'].apply(transform_node_value)

    # Create a list of tuples in the desired format
    tuples_list = [((int(row['node_1']), int(row['node_2'])), int(row['start_of_interaction'])) for _, row in df.iterrows()]

    file_path = file_path.replace('.csv', "")
    # Write the tuples to the output txt file
    with open(file_path+"-TEXT.txt", 'w') as file:
        for item in tuples_list:
            file.write(f"{item[0][0]} {item[0][1]} {item[1]}\n")

    return tuples_list

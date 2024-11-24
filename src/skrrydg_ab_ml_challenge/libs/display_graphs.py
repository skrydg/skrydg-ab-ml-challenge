import matplotlib.pyplot as plt

def display_graphs(display_functions, column_count=5):
    row_count = (len(display_functions) + column_count - 1) // column_count
    figure, axis = plt.subplots(row_count, column_count, figsize=(3 * column_count, 3 * row_count))
    figure.tight_layout()
    for index, display_functions in enumerate(display_functions):
        display_functions(axis[index // column_count, index % column_count])
    plt.show()

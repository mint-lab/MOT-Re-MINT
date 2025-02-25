import numpy as np
import matplotlib.pyplot as plt

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def draw_heatmap(data, row_labels, col_labels):
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='coolwarm')

    ax.set_xticklabels([''] + col_labels)
    ax.set_yticklabels([''] + row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, round(data[i, j], 2), ha='center', va='center', color='black')

    fig.colorbar(cax)
    plt.show()

def test_draw_heatmap():    
    data = np.random.rand(4, 4)
    row_labels = ['A', 'B', 'C', 'D']
    col_labels = ['W', 'X', 'Y', 'Z']
    draw_heatmap(data, row_labels, col_labels)

if __name__ == "__main__":  
    test_draw_heatmap()
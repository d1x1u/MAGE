import numpy as np
from scipy import io
from matplotlib import pyplot as plt

def get_palette(dataset_name):
    if dataset_name == "Trento":
        palette = {
            0: (0, 0, 0), # background
            1: (0, 255, 0), # Apple trees
            2: (178, 24, 43), # Buildings
            3: (255, 255, 0), # Ground
            4: (139, 69, 19), # Woods
            5: (0, 100, 0), # Vineyard
            6: (0, 0, 255) # Roads
        }
    if dataset_name == "MUUFL":
        palette = {
            0: (0, 0, 0), # background
            1: (0, 100, 0), # Trees
            2: (0, 255, 0), # Mostly grass
            3: (0, 255, 255), # Mixed ground surface
            4: (255, 165, 0), # Dirt and sand
            5: (255, 0, 0), # Road
            6: (0, 0, 255), # Water
            7: (128, 0, 128), # Building shadow
            8: (255, 182, 193), # Buildings
            9: (160, 82, 45), # Sidewalk
            10: (255, 255, 0), # Yellow curb
            11: (139, 0, 0), # Cloth panels
        }
    if dataset_name == "Houston":
        palette = {
            0: (0, 0, 0), # background
            1: (50, 205, 50), # Healthy grass
            2: (0, 255, 0), # Stressed grass
            3: (0, 128, 128), # Synthetic grass
            4: (0, 100, 0), # Trees
            5: (139, 69, 19), # Soil
            6: (0, 0, 139), # Water
            7: (255, 255, 255), # Residential
            8: (255, 255, 0), # Commercial
            9: (128, 128, 128), # Road
            10: (128, 0, 0), # Highway
            11: (255, 20, 147), # Railway
            12: (219, 113, 147), # Parking Lot 1
            13: (205, 133, 63), # Parking Lot 2
            14: (255, 0, 255), # Tennis Court
            15: (0, 255, 255), # Running Track
        }
    return palette

def convert_to_color_(arr_2d, dataset_name):
    """Convert an array of labels to RGB color-encoded image.
    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)
    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    if dataset_name is None or dataset_name not in ["MUUFL", "Trento", "Houston"]:
        raise ValueError("Dataset must be one of MUUFL, Trento, Houston")
    
    palette = get_palette(dataset_name)

    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def visualize_(arr_2d, dataset_name, path):
    plt.imshow(convert_to_color_(arr_2d, dataset_name))
    plt.axis("off")
    plt.savefig(path)

if __name__ == "__main__":
    gt = io.loadmat("../dataset/Trento/processed/gt.mat")['gt']
    TRLabel = io.loadmat("../dataset/Trento/processed/TRLabel.mat")['TRLabel']
    TSLabel = io.loadmat("../dataset/Trento/processed/TSLabel.mat")['TSLabel']

    visualize_(gt, "Trento", "../visualize/Trento/gt.png")
    visualize_(TRLabel, "Trento", "../visualize/Trento/TR.png")
    visualize_(TSLabel, "Trento", "../visualize/Trento/TS.png")

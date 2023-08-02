import numpy as np
import matplotlib.pyplot as plt

def compare_results(outputs, ground_truth):
    """
    Plot two sets of images side by side.

    Parameters:
        outputs (numpy.ndarray): An array of images to be displayed on the left side.
                                 Shape should be (num_images, height, width, channels).
        ground_truth (numpy.ndarray): An array of images to be displayed on the right side.
                                      Shape should be (num_images, height, width, channels).

    Raises:
        ValueError: If the shapes of the two input arrays are not compatible.

    Returns:
        None
    """
    outputs = outputs.detach().numpy()
    ground_truth = ground_truth.detach().numpy()
    if outputs.shape != ground_truth.shape:
        raise ValueError("Input arrays should have the same shape.")

    num_images = outputs.shape[0]
    fig, ax = plt.subplots(num_images, 2, figsize=(10, 5*num_images))

    for i in range(num_images):
        ax[i, 0].imshow(outputs[i])
        ax[i, 0].axis('off')
        ax[i, 0].set_title(f"Output {i+1}")

        ax[i, 1].imshow(ground_truth[i])
        ax[i, 1].axis('off')
        ax[i, 1].set_title(f"Ground Truth {i+1}")

    plt.tight_layout()
    plt.show()
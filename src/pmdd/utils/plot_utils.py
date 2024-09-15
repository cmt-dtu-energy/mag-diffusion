import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def plot_magfield(field: np.ndarray, vmax: float = 1) -> None:
    plt.clf()
    labels = ["Bx-field", "By-field", "Bz-field"]
    nrows = 3 if len(field.shape) == 4 else 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=3, sharex=True, sharey=True, figsize=(15, 10)
    )
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    if len(field.shape) == 3:
        for i, comp in enumerate(field):
            ax = axes.flat[i]
            im = ax.imshow(comp, cmap="bwr", norm=norm, origin="lower")
            ax.set_title(labels[i])

    elif len(field.shape) == 4:
        for i, z in enumerate([0, 1, 2]):
            for j, comp in enumerate(field[:, :, :, z]):
                ax = axes.flat[i * 3 + j]
                im = ax.imshow(comp, cmap="bwr", norm=norm, origin="lower")
                ax.set_title(labels[j] + f"@{z+1}")

    else:
        raise NotImplementedError

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.345, 0.015, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
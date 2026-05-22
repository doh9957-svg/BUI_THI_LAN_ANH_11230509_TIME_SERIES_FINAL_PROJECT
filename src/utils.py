import matplotlib.pyplot as plt
from config import PLOTS_DIR


def save_plot(filename: str, show: bool = False):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()
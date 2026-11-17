from datetime import datetime
import os

def _save_plot(save_dir, filename, plots):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, stamp)
    os.makedirs(save_dir, exist_ok=True)

    save_plot_name = os.path.join(save_dir, f"{filename}.png")

    plots.savefig(save_plot_name, dpi=150, bbox_inches="tight")

    return save_plot_name
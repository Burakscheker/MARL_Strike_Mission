import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from config import (
    GOAL,
    GRID_MAX,
    GRID_MIN,
    INNER_HALF_SIZE,
    OUTER_HALF_SIZE,
    RADAR_CENTERS,
    START,
)


def plot_map(output, trajectories=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 10))
    axis.set_facecolor("#081018")
    figure.patch.set_facecolor("#081018")

    for coordinate in range(GRID_MIN, GRID_MAX + 1, 100):
        axis.axhline(coordinate, color="#24483b", linewidth=0.5, alpha=0.65)
        axis.axvline(coordinate, color="#24483b", linewidth=0.5, alpha=0.65)

    for index, (center_x, center_y) in enumerate(RADAR_CENTERS, start=1):
        axis.add_patch(
            Rectangle(
                (center_x - OUTER_HALF_SIZE, center_y - OUTER_HALF_SIZE),
                OUTER_HALF_SIZE * 2,
                OUTER_HALF_SIZE * 2,
                facecolor="#f59e0b",
                edgecolor="#f59e0b",
                linewidth=2,
                alpha=0.16,
            )
        )
        axis.add_patch(
            Rectangle(
                (center_x - INNER_HALF_SIZE, center_y - INNER_HALF_SIZE),
                INNER_HALF_SIZE * 2,
                INNER_HALF_SIZE * 2,
                facecolor="#ef4444",
                edgecolor="#ef4444",
                linewidth=2,
                alpha=0.20,
            )
        )
        axis.plot(center_x, center_y, "+", color="#d1fae5", markersize=13, markeredgewidth=2)
        axis.text(
            center_x + 14,
            center_y - 24,
            "R{} ({}, {})".format(index, center_x, center_y),
            color="#d1fae5",
            fontsize=9,
        )

    if trajectories:
        colors = ("#38bdf8", "#a78bfa")
        for index, route in enumerate(trajectories):
            if not route:
                continue
            x_values, y_values = zip(*route)
            axis.plot(
                x_values,
                y_values,
                color=colors[index % len(colors)],
                linewidth=2,
                label="Uçak {} rotası".format(index + 1),
                zorder=5,
            )

    axis.scatter(*START, s=120, color="#38bdf8", edgecolor="white", zorder=6)
    axis.text(START[0] + 18, START[1] - 32, "B ×2 {}".format(START), color="#38bdf8")
    axis.scatter(*GOAL, s=210, marker="*", color="#fb7185", edgecolor="white", zorder=6)
    axis.text(GOAL[0] - 190, GOAL[1] + 26, "H {}".format(GOAL), color="#fb7185")

    axis.set_xlim(GRID_MIN, GRID_MAX)
    axis.set_ylim(GRID_MIN, GRID_MAX)
    axis.set_aspect("equal")
    axis.set_xlabel("x", color="#9ca3af")
    axis.set_ylabel("y", color="#9ca3af")
    axis.set_title(
        "Strike Mission — 1000×1000 grid, giriş-bazlı radar riski",
        color="#d1fae5",
    )
    axis.tick_params(colors="#9ca3af")
    for spine in axis.spines.values():
        spine.set_color("#3f8067")
    if trajectories:
        axis.legend(loc="lower left", framealpha=0.25)
    figure.savefig(output, dpi=140, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the Strike Mission map")
    parser.add_argument("--output", default="runs/map.png")
    args = parser.parse_args(argv)
    print(plot_map(args.output))


if __name__ == "__main__":
    main()

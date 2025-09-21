# 📊 SCRIPT 3: Bar plot of number of videos by FPS
import matplotlib.pyplot as plt

fps_values = df["fps"].value_counts().sort_index()

plt.figure(figsize=(10, 6))
bars = plt.bar(fps_values.index.astype(str), fps_values.values, color="skyblue", edgecolor="black")
plt.xlabel("FPS", fontsize=12)
plt.ylabel("Number of Clips", fontsize=12)
plt.title("Number of Videos by FPS", fontsize=14)
plt.xticks(rotation=0)

for bar, count in zip(bars, fps_values.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(count),
             ha="center", va="bottom", fontsize=10)

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
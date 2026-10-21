import json
import matplotlib.pyplot as plt
from models import get_models
from cost import measure_latency

with open("results.json") as f:
    results = json.load(f)

models = get_models()
lat1, lat128 = {}, {}
for name, m in models.items():
    lat1[name] = measure_latency(m, 1, 100)
    lat128[name] = measure_latency(m, 128, 20)
    print(f"{name:10s} batch=1: {lat1[name]:.3f} ms   batch=128: {lat128[name]:.3f} ms")

colors = {"MLP": "tab:blue", "CNN": "tab:orange", "Attention": "tab:green"}
names = list(results.keys())


def scatter(ax, xs, xlabel, title, logx=False):
    for n in names:
        acc = results[n]["test_acc"] * 100
        ax.scatter(xs[n], acc, s=120, color=colors[n], label=n)
        ax.annotate(n, (xs[n], acc), textcoords="offset points", xytext=(8, 6))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
params = {n: results[n]["params"] for n in names}
scatter(axes[0], params, "Parameters", "Accuracy vs parameters", logx=True)
scatter(axes[1], lat1, "Latency, batch=1 (ms)", "Accuracy vs latency (batch 1)")
scatter(axes[2], lat128, "Latency, batch=128 (ms)", "Accuracy vs latency (batch 128)")
plt.tight_layout()
plt.savefig("tradeoff.png", dpi=150)
print("Saved tradeoff.png")
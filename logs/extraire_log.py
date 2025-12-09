import matplotlib.pyplot as plt

epochs = list(range(1, 31))

train_acc = [11.76, 15.66, 17.68, 21.04, 23.43, 25.56, 27.12, 29.09, 31.01,
             32.52, 34.93, 35.70, 37.13, 38.66, 39.34, 40.57, 41.68, 42.46,
             43.33, 43.73, 44.82, 45.28, 45.92, 46.35, 46.55, 46.77, 47.73,
             48.43, 48.34, 48.87]

dev_acc = [11.46, 15.14, 17.04, 20.65, 22.52, 24.36, 26.07, 27.74, 29.23,
           30.75, 32.65, 33.05, 34.27, 35.66, 36.56, 37.13, 38.19, 38.74,
           39.00, 39.63, 40.40, 40.30, 40.70, 40.94, 41.21, 41.31, 41.69,
           42.28, 41.79, 42.14]

plt.figure(figsize=(9,5))
plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
plt.plot(epochs, dev_acc, label="Validation Accuracy", marker='s')

plt.title("GAP CNN – Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("gap_cnn_accuracy.png", dpi=300)
plt.show()
loss = [3.5183, 3.1891, 3.0224, 2.8910, 2.7776, 2.6879, 2.6133, 2.5494,
        2.4957, 2.4414, 2.3966, 2.3561, 2.3186, 2.2853, 2.2503, 2.2256,
        2.1983, 2.1737, 2.1493, 2.1289, 2.1099, 2.0912, 2.0788, 2.0563,
        2.0450, 2.0321, 2.0157, 2.0006, 1.9928, 1.9805]

plt.figure(figsize=(9,5))
plt.plot(epochs, loss, marker='o', color='red')

plt.title("GAP CNN – Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("gap_cnn_loss.png", dpi=300)
plt.show()

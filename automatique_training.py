import subprocess
import os

learning_rates = [0.01, 0.005, 0.001, 0.0001]

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

batch_size = 100

for lr in learning_rates:
    lr_str = str(lr).replace('.', '_')
    log_path = os.path.join(log_dir, f"log_lr_{lr_str}.txt")

    print(f" Lancement entraînement : lr={lr}  | log → {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        # Appel de training_One2One.py
        subprocess.run(
            [
                "python",
                "training_One2One.py",
                "--lr", str(lr),
                "--batch_size", str(batch_size)
            ],
            stdout=f,               # redirige stdout vers le fichier log
            stderr=subprocess.STDOUT,  # redirige aussi les erreurs dedans
            check=False
        )

    print(f" Terminé pour lr={lr}")

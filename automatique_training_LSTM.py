import subprocess
import os

learning_rates = [0.0001, 0.01, 0.1] 
batch_size_fixed = 1000  

log_dir = "logs"  
os.makedirs(log_dir, exist_ok=True)

for lr in learning_rates:
    lr_str = str(lr).replace('.', '_')  
    log_path = os.path.join(log_dir, f"log_lr_{lr_str}_batch_{batch_size_fixed}.txt") 

    print(f"Launching training with lr={lr} and batch_size={batch_size_fixed} | log → {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(
            [
                "python",
                "training_Many2One.py",
                "--lr", str(lr),
                "--batch_size", str(batch_size_fixed)
            ],
            stdout=f,               
            stderr=subprocess.STDOUT,  
            check=False
        )

    print(f"Training completed for lr={lr} and batch_size={batch_size_fixed}")

lr_fixed = 0.005  
batch_sizes = [500, 200, 100] 

for batch in batch_sizes:
    batch_str = str(batch)  
    log_path = os.path.join(log_dir, f"log_lr_{lr_fixed}_batch_{batch_str}.txt")  # Path for the log file

    print(f"Launching training with lr={lr_fixed} and batch_size={batch} | log → {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(
            [
                "python",
                "training_Many2One.py",
                "--lr", str(lr_fixed),
                "--batch_size", str(batch)
            ],
            stdout=f,               # Redirect stdout to the log file
            stderr=subprocess.STDOUT,  # Redirect stderr to the same log file
            check=False
        )

    print(f"Training completed for lr={lr_fixed} and batch_size={batch}")

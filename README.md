# DeepLearning2PlayOthello


## Installation

To install the required dependencies, run the following command:

```bash
pip install scikit-learn pandas tabulate matplotlib h5py scipy tqdm torch torchvision
```
Or
```bash
pip install -r requirements.txt
```

This package contains files as follows:

- **networks_[your_student_number].py**: This file contains your networks. You can define and compare several models with different class names in this file. You should change the name of this file to your student number, which is unique.
- **training_[x].py**: This code is provided as an example training script.
- **data.py**: This file is responsible for handling data loading and preprocessing tasks. It includes functions to prepare datasets for training and evaluation, ensuring compatibility with the models defined in the networks file.
- **game.py**: This code is used for playing the Othello game between two models. The game will be run twice with different colors for each player (different starting player), and it generates a GIF file as the log of each game.
- **utils.py**: This file contains functions that will be used as the rules and different steps of the games.
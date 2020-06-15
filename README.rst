"Guided Image Generation with Conditional Invertible Neural Networks" (2019)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Paper: https://arxiv.org/abs/1907.02392
Supplement: https://drive.google.com/file/d/1_OoiIGhLeVJGaZFeBt0OWOq8ZCtiI7li

Contents
^^^^^^^^^^^^^^^^

Each subdirectory has its own README file containing the details.

* ``experiments/mnist_minimal_example`` contains code to produce class-conditional MNIST samples in <150 lines total
* ``experiments/colorization_minimal_example`` contains code to colorize LSUN bedrooms in <200 lines total
* ``experiments/colorization_cINN`` contains the full research code used to produce all colorization figures in the paper
* ``experiments/mnist_cINN`` contains the full research code used to produce all mnist figures in the paper

Dependencies
^^^^^^^^^^^^^^^^

Except for pytorch, any fairly recent version will probably work, 
these are just the confirmed ones:

+---------------------------+-------------------------------+
| **Package**               | **Version**                   |
+---------------------------+-------------------------------+
| Pytorch                   | 1.0.X                         |
+---------------------------+-------------------------------+
| tqdm                      | >= 4.39.0                     |
+---------------------------+-------------------------------+
| Numpy                     | >= 1.15.0                     |
+---------------------------+-------------------------------+
| FrEIA                     | >= 0.2.0                      |
+---------------------------+-------------------------------+
| *Optionally for the experiments:*                         |
+---------------------------+-------------------------------+
| Matplotlib                | 2.2.3                         |
+---------------------------+-------------------------------+
| Visdom                    | 0.1.8.5                       |
+---------------------------+-------------------------------+
| Torchvision               | 0.2.1                         |
+---------------------------+-------------------------------+
| scikit-learn              | 0.20.3                        |
+---------------------------+-------------------------------+
| scikit-image              | 0.14.2                        |
+---------------------------+-------------------------------+
| Pillow                    | 6.0.0                         |
+---------------------------+-------------------------------+

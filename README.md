# IDATT2900

Bachelor Thesis - Gaussian-Driven Model-Based Reinforcement Learning with Vector Quantisation for State Generalisation

## Installation Guide

### Python Version

This repository has been developed and tested using **Python 3.12.5** and **Python 3.11.8**. While it may operate on other Python 3.x versions, compatibility with versions other than **Python 3.12.5** and **Python 3.11.8** **is not guaranteed**.  
Please ensure that Python is already installed on your system before proceeding.

---

### Step 1: Clone the Repository

To obtain the source code, clone this repository using the following command in your terminal or command prompt:

```bash
git clone https://github.com/EmilJohns1/BachelorThesis.git
cd BachelorThesis
```

### Step 2: Download Packages

```bash
pip install -r requirements.txt
```

### Step 3: Execute Terminal Line Commands

Run the program with command-line arguments specified in `main.py`. For example:

```bash
python main.py --agent model-based --env CartPole-v1 --training_time 100
```

### Additional Information

- The `requirements.txt` file has been tested with the `CartPole-v1` environment.
- For environments that require `Box2D`, `SWIG` is a prerequisite. See [SWIG Official Website](http://www.swig.org/) for installation instructions.
- To install Box2D support, run: `pip install "gymnasium[box2d]"`

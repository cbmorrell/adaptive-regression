# CIIL for Regression

## Publication

**Authors:** Christian Morrell, Evan Campbell, Ethan Eddy, and Erik Scheme

This repository hosts code for the publication: **TODO**. Here we investigate context-informed incremental learning (CIIL) for regression-based myoelectric control. Please cite this publication if using this for research purposes. For any questions related to the work or this repository, please reach out to <cmorrell@unb.ca>

## Usage

This repository contains two primary scripts:

- `main.py` - Used for running the experiment, including data collection and target acquisition.
- `analyze.py` - Used to analyze data after running an experiment.

Both scripts provide a command-line interface (CLI), described in detail below.

### `main.py`

The CLI for `main.py` supports two main objectives: configuring participant settings and running the experiment.

```bash
python main.py <objective> [arguments]
```

#### 1. **Configuration (`config`)**

Creates a configuration file for a participant, specifying their information and experimental setup. This will create a `participant.json` file (printed in the console), which you then use when running the experiment.

**Usage:**

```bash
python main.py config <subject_directory> <device> <dominant_hand> <age> <sex> <experience>
```  

**Arguments:**

- `subject_directory` (str): Directory where participant data will be stored. The directory's name is used as the subject ID.  
- `device` (str): EMG device to be used. Choices: `emager`, `myo`, `oymotion`, `sifi`.  
- `dominant_hand` (str): Participant's dominant hand. Choices: `left`, `right`.  
- `age` (int): Participant's age.  
- `sex` (str): Participant's sex.  
- `experience` (str): Level of experience with myoelectric control. Choices:  
  - `N`: Novice  
  - `I`: Intermediate  
  - `E`: Expert  

**Example:**

```bash
python main.py config data/subject-001 myo right 25 M N
```

#### 2. **Running the Experiment (`run`)**

Collects EMG data for a specific condition and stage of the experiment. The order of the models is based on a latin square. Stages typically go from SGT, to adaptation, to validation. For non-adaptive models, adaptation is not required (and will throw an error if selected).

**Usage:**

```bash
python main.py run <participant> <condition_idx> <stage> [--analyze]
```  

**Arguments:**

- `participant` (str): Path to the participant's `participant.json` file.  
- `condition_idx` (int): Index of the current condition (starting from 0).  
- `stage` (str): Experimental stage. Choices:  
  - `sgt`: Screen-guided training  
  - `adaptation`: Adaptive phase  
  - `validation`: Validation phase

**Example:**

```bash
python main.py run data/subject-001/participant.json 0 adaptation
```

### `analyze.py`

The CLI for `analyze.py` changes the format of resulting figures.

**Usage:**

```bash
python analyze.py [options]
```

**Arguments:**

- `-p, --participants` (default: `'all'`): Specify a participant or a list of participants to evaluate. By default, all participants are included.
- `-pl, --presentation_layout` (flag): Enables a presentation-friendly plotting style. If not set, the script generates plots in a report format.

### Utility Functions

Although not needed to run the experiment, some of the logic, such as neural network creation and adaptation, are handled in utility functions found in the `utils` directory. These functions are used throughout the main scripts, and are broken up into different modules for convenience.

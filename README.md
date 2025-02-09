# Text Classifier for Emotion Recognition v2.0

#### Author
- Wilfredo Graterol

#### Revision by
- Arturo yepez

# Description

For social robots, knowledge regarding human emotional states is an essential part of adapting their behavior or associating emotions to other entities. They can gather the information from which emotion detection is processed via different media, such as text, speech, images, or videos.

A framework is proposed to allow social robots to detect emotion and to store this information in semantic repositories, based on EMONTO (an EMotion ONTOlogy).

As a proof-of-concept, it was developed a first version of this framework focused on emotion detection in text, which can be obtained directly as text or by converting speech to text.

Currently, the first version was revised into the current implementation were the main focus was to made the process of training more dynamic to allow experimentation or changes in the models/databases without too much effort.

## Architecture

According to the multimedia data considered, a wide range of emotion detection techniques can be implemented, mostly based on machine learning models.

The emotion detection task was approached as a multi-label classification problem, modeled with a Transformer architecture and meta-learning. Eleven emotions were considered: *anger*, *anticipation*, *disgust*, *fear*, *joy*, *love*, *optimism*, *pessimism*, *sadness*, *surprise*, and *trust*.

The classification task consisted of labeling a text as "*neutral or no emotion*" or as one or more of the previously mentioned emotioned, i.e., a binary vector indicating if each emotion was detected (1) or not (0).

Using the previously mentioned Transformers, a stacked ensemble architecture was designed:
* Each *Weak Learner* consists of a transformer, used to create a sentence embedding and a Multi-Layer Perceptron (MLP) with two hidden layers.

![Weak-Learner architecture](./assets/img/weak-learner.png)

* The Meta-Learning consists of an mLP with one hidden layer.

![Meta-Learner architecture](./assets/img/meta-learner.png)


# Setup

## Python Environment

First, we have to make sure to have a Python environment for development or testing since its the recommended way to ensure the consistency when working with many different environments.

In order to create a new environment we use the following command:
```bash
python3 -m venv ./venv
```

After that, we enter the environment with the following command from the root of the repository:
```bash
# Linux / Unix
source venv/bin/activate

# Windows
./venv/Scripts/activate
```

Finally, we install the recommended packages with the following command:
```bash
python3 -m pip install -r requirements.txt
```

When we finish to work on the project, let's make sure to deactivate the environment with the following command:
```bash
deactivate
```

## Models for Sentence Transformers

If it's the first time running the script, most likely you'll need to download the models that are going to be used. This process will be done automatically by the library that we are using, `sentence-transformers`.

They are stored in: `~/.cache/huggingface/`

# Use

## Commands available

The main script that contains the main functionality is `main.py` and it has the following commands available:

### Train the model
For this option, we go through the whole process of training the Weak Learners and the Meta-Learner with the data provided as parameters in the script. At the end of the training process, all learners are saved in the `out/` directory.

* Default value for this option is `False`

```bash
python3 main.py --train
# or
python3 main.py -t
```

### Load a model
This option allows us to load a model that was previously trained and stored in the `models` directory. Usually used when we want to make predictions with a model that was already trained or get to know the performance of the model  (calculating the metric associated).

* Receives a parameter with the name of the model to load, if no model is provided, an error will be raised.
* Default value for this option is `""`

```bash
python3 main.py --load {{ model_name }}
# or
python3 main.py -l {{ model_name}}
```

An error will be raised if the model is not found in the `out/` directory with the `.lnr` or `clf` file extension.

### Predict
This option allows us to make predictions with a model of the Classifier (either by training or loading a model). The input data is provided as a parameter in the script.

Currently we **only** support files compatible with the `load_csv` function from the `pandas` library. 

* Receives a parameter with the path to the file with the text to predict, if no path is provided, an error will be raised.
* Default value for this option is `""`
* If we don't specifiy training or loading a model, the script will raise an error.

```bash
python3 main.py --predict {{ path_to_file }}
# or
python3 main.py -pr {{ path_to_file }}
```

Proper prediction should be a CSV file with the following format:
```csv
text
{{ line 1 }}
{{ line 2 }}
...
{{ line n }}
```

### Score
This option allows us to calculate different metric scores for the model provided as a parameter. 

The metrics calculated are the following:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

* Receives a parameter with the name of the model to load, if no model is provided, an error will be raised.
* Default value for this option is `False`

```bash
python3 main.py --score
# or
python3 main.py -s
```

### Params path
Project supports custom parameters for many of the different parts of the script. By default, some pre-configured parameters are provided in the `parameters.default.json` file.

* Default value for this option is `paramaters.default.json`

```bash
python3 main.py --params {{ path_to_file }}
# or
python3 main.py -p {{ path_to_file }}
```

### Verbose execution
This option allows us to see the output of the script in a more verbose way. By default, the script only shows the most important information.

* Default value for this option is `False`

```bash
python3 main.py --verbose
# or
python3 main.py -v
```
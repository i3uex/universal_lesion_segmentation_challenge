# MEDICAL IMAGE SEGMENTATION
## Introduction
This project is a part of a final degree project in Software Engineering at the University of Extremadura. The main goal of the project is to develop a tool that allows the user to segment medical images using deep learning techniques. The dataset used in this project is related to COVID-19 infection segmentation.

The full implemented pipeline of the project is show in the following picture:

![Pipeline](./Pipeline_TFG.png)

As shown, the pipeline is divided into three main parts:
1. Datasets used
2. Pipeline (Data Preprocessing, Model Training, Model Evaluation)
3. Results creation

# Execution
First of all, you need to create a virtual environment and install the requirements. To do this, you can execute the following commands:

```bash
pip install -r requirements.txt
```

In case you are using conda, you can create the environment using the following command:

```bash
conda create -f environment.yml
```

After that, you can execute the script to download the datasets from the root of the project:

```bash
./scripts/script.sh
```

When the datasets are downloaded, you can execute the script to normalize the spacing of the images:

```bash
python preprocess_spacing.py
```

Finally, you can execute the main script to train the model, providing the desired parameters:

```bash
python main.py --epochs=3000 --architecture=unetr --loss=dice
```



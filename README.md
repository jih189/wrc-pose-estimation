# wrc-pose-estimation
==============================
### Description
Pose estimation of WRC objects. The pose estimation is splited into four sections.
1. YOLO
    we use yolo to detect the object in the image, and crop the object from the image.
2. Rot-classifier
    this classifier will estimate the rough pose of the object.
3. PSPNet
    from the crop image, it predicts the mask of the object in the image.
4. RefineNet
    With the crop image, rendered mask-edge, and the predicted mask, this net will refine the pose estimation.

### Instruction

#### Setup

python setup.py install
Modify file src/configuration.py

#### YOLO setup

#### Collect image and pose

python src/data/getLabelFromImage.py

This is a script to help you to collect the image with the object pose. Make the object in the image to align with the
rendered red edge, and press 'c' to save the image. Press 'q' to quit the program. Update the datapoints if need. 

#### Rot-classifier



##### Author: Jiaming Hu

##### project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── config.py          <- configuration file for pose estimation
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   ├── *_rot      <- The data used for rot-classifier
    │   │   └── *_refine   <- The data used for pose refinement net and PSP net
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── model.py       <- Rot-classifier, RefineNet, and PSPNet
    │   └── models.py      <- Yolo model
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py              <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                   <- Source code for use in this project.
    │   ├── __init__.py       <- Makes src a Python module
    │   ├── common            <- Helping functions
    │   │   ├── DataLoader.py <- Data loaders
    │   │   ├── object_model.py
    │   │   ├── fscore.py
    │   │   └── chamfer2D     <- chamfer distance losss for 2d
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── getLabelFromImage.py
    │   │   ├── make_rot_dataset.py
    │   │   └── make_refine_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── poseEstimation.py      <- Running whole pipeline of pose estimation
    │   │   ├── predict_psp_model.py   <- PSPNet test
    │   │   ├── precit_refine_model.py <- RefineNet test
    │   │   ├── predict_rot_model.py   <- rot-classifier test
    │   │   ├── train_psp_model.py     <- PSPNet train file
    │   │   ├── train_refine_model.py  <- RefineNet train file
    │   │   └── train_rot_model.py     <- rot-classifier train file
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
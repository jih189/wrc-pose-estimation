# wrc-pose-estimation

==============================
### Description
Pose estimation of WRC objects. The pose estimation is splited into four sections.
1. YOLO:<br/>
    we use yolo to detect the object in the image, and crop the object from the image.
2. Rot-classifier:<br/>
    this classifier will estimate the rough pose of the object.
3. RefineNet:<br/>
    With the crop image, rendered mask-edge, and the predicted mask, this net will refine the pose estimation.

### Instruction

#### YOLO setup

For object detection, we use the work which has done a good job, so we need to train this part in another directory.<br/>
To begin:<br/>
Go to link https://github.com/linghongyi/yolov3, git clone this project.<br/>
Install python 3.7 or later version.<br/>
Go to the root dir of yolov3 and run
```
pip install -U -r requirements.txt
```
Training YOLO:<br/>
Before training yolo, make sure the following steps have been done:<br/>
1.    Put all images of the dataset under path: yolov3/data/images/<br/>
2.    Put all labels of the dataset under path: yolov3/data/labels/. REMEMBER: the file name of the label should be the same as the corresponding image. For example, the label file of image 000.<br/>jpg should be 000.txt<br/>
3.    Create a file named wrs.data under path: yolov3/data/<br/>
      The file should contains following lines:<br/>
```
classes=<number of object class>
train=data/train.txt
valid=data/validation.txt
names=data/wrs.names
```
4.   Create a file named wrs.names under path: yolov3/data/<br/>
      The file should contain the name of all the objects. Each line is a name. REMEMBER: the order of objects should be the same as the index when you do the labeling. For example<br/>
```
obj1
obj2
obj3
```
1.   Split the dataset into a training set and validation set. Randomly select 80% of the dataset to be training set and the rest is validation set. After random sampling, there are two steps to do:<br/>
 
       i. First, Create a file named train.txt under path: yolov3/data/<br/>
      This file should contain the path of all the training images. For example, “data/images/000000.jpg”. Each line is a path of a training image.<br/>
      ii. Next, Create a file named validation.txt under path: yolov3/data/<br/>
      This file should contain the path of all the validation images. For example, “data/images/000000.jpg”. Each line is a path of a validation image.<br/>
 
2.   The network architecture we are using is yolo3-tiny3. Open yolo3/cfg/yolov3-tiny3.cfg. There are three YOLO blocks in this file, they looks like:<br/>
```
            [yolo]
            mask = 0,1,2<br/>
            #anchors =  30,30,  62,45,  59,119,  116,90,  156,198,  373,326,  450,450,  400,500, 550,600
            anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
            classes=5        # make sure this number equals to number of classes
            num=9
            jitter=.3
            ignore_thresh = .7
            truth_thresh = 1
            random=1
```
Previous block of these three YOLO block are convolutional blocks, they looks like:<br/>
```
            [convolutional]
            size=1
            stride=1
            pad=1
            filters=30         # make sure this number equals to 3 * (number of classes + 5)
            activation=linear
```
 
All these steps are done, using command<br/>
```
python train.py --data data/wrs.data --cfg cfg/yolov3-tiny3.cfg --weights weights/yolov3-tiny.conv.15 --epochs <number of epochs to train, usually 200 or more>
```
 
Test YOLO:<br/>
Put the test images under path: yolov3/data/samples, the output will be in yolov3/output<br/>
 
Command is 
```
python detect.py --names data/wrs.names --cfg cfg/yolov3-tiny3.cfg --weights weights/best.pt --conf-thres <confidence threshold, from 0 to 1, default is 0.3> --iou-thres <IOU threshold using for non-maximum suppression, from 0 to 1, default 0.6>
```
#### Collect image and pose
```
python src/data/getLabelFromImage.py
```
This is a script to help you to collect the image with the object pose. Make the object in the image to align with the
rendered red edge, and press 'c' to save the image. Press 'q' to quit the program. Update the datapoints if need.<br/><br/><br/>

----------------


## training rough pose estimation and refinement
We recommend you run everything in conda because the requirements of the environment of following is different from the environment of yolo. After you create a environment, <br/>
Here is the command to install conda <br/>
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# follow line is create a new conda environment
conda create -n newenv
```
Run <br/>
```
pip3 install --upgrade pip
pip install -r requirements.txt
```
under the project root, so it will install packages you need. If you have run it before, then please skip it.<br/><br/>

At the beginning, we need to create a directory called “data” which contains 4 directories “images”, “mesh”, “processed”, and “raw”. “Images” is the file constraining the image and pose which are collected from the getLabelFromImage.py. That is, after image-pose pairs are collected, then they will be saved in this directory. “Mesh” is the directory containing the object description file like .obj file. “Raw” is the directory which contains the verified image-pose pairs. Because some pose of the object in image may not be accurate, we use the labelverifier.py to remove the fault pair, and save the proper pair into the “raw”. “Processed” is the directory containing the pre-processed data for rough pose estimation training and pose refinement training.<br/><br/>

For training the model for an object, we need to create directories to store the object data.<br/>
Assuming we want to train the pulley, we need to create a directory named “pulley” in both “images'' and “raw” directories, and put all poses and images of the pulley into “data/images/pulley”. Furthermore, you need to save the object model to the “mesh”. For example, the object model of pulley is MBRFA30-2-P6.obj, then you need to save it into the “mesh” directory. In the “processed” directory, you need to create “pulley_iterative_refine”, “pulley_refine”, and “pulley_rot” directories.<br/><br/>

As the conclusion, for the object pulley, the data directory will be like following:<br/>
```
data
├── images
|    └── pulley
├── mesh
|    ├── MBRFA30-2-P6.obj
|    └── MBRFA30-2-P6.mtl(this is optional. If you do not want it, then delete line “mtllib” in .obj)
├── raw
|    └── pulley
└── processed
       ├── pulley_refine
       ├── pulley_iterative_refine
       └── pulley_rot
```

After that, you need to update the configuration.py for your object:
```
OBJ_NAME = "pulley"
CAD_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"
SAMPLE_FACE_MODEL = CURRENT_POSE_ESITMATION_DIR + "data/mesh/MBRFA30-2-P6.obj"
```

And you also need to provide the camera information:
```

CAMERA_MATRIX = np.array(
    [
        [1390.6298269250192, 0, 665.4334864497848],
        [0, 1389.3521948493603, 314.5310503226418],
        [0, 0, 1],
    ],
    dtype="double",
)

CAMERA_W = 1280
CAMERA_H = 720
```


Everytime you change the configuration.py, please run
```
python setup.py install
```
For verify the image pose pairs, run
```
python src/data/labelverifier.py
```
This script will render the object based on the pose of the object onto the image, so you can check whether the pose is right or not. In the upper left corner of the window, it shows the current image index and the “is pose correct”. If it is true, then this image-pose pair will be saved to “raw/pulley”. If it is false, it will not be saved later. Moveover, there are five keys：<br/>
```
    S key: save all image-pose pairs whose “is pose correct” flag is true to the raw directory
    M key: next image-pose pair
    N key: last image-pose pair
    C key: switch the “is pose correct” flag of current image-pose pair
    Q key: quit the script without save
```
Therefore, you need to set all fault image-pose pair’s “is pose correct” to false, and then press ‘s’ to save the data.<br/><br/>

After that, we remove fault data from the dataset, then we need to preprocess the dataset for data augmentation. Run following commands<br/>
```
python src/data/make_rot_dataset.py
python src/data/make_refine_dataset.py
python src/data/make_iterative_refine_dataset.py
```

At this point, the pre-processed are stored in “pulley_iterative_refine”, “pulley_refine”, and “pulley_rot” directories. Then, we can train the rot model and refinement model.<br/>
First, run following command to train rot model<br/>
```
python src/models/train_rot_model.py
```
For running refinement training, we need to train the flownet first, so run <br/>
```
python src/model/train_flow_model.py
```
To pre-train the flownet.<br/>
Then run<br/>
```
python src/models/train_refine_model_iterative.py
```
To train the refinement.<br/><br/>

While you are training the refinement model, please pay attention to the add score. If it does not increase for a long time, then you have to tune the model. Furthermore, the training time could take several days.<br/><br/>

For testing the model<br/>
1. Ensure all weights of the model like “best_model_iterative_refine_(object_name).pth”, “best_model_rot_(object_name).pth”, and “best_yolo_model_wrc.pt” into the “weights” directory under the project root.<br/><br/>

2. In the “data” directory, copy the wrs-wrc.names(you can download it from the wrc-data folder) file there. This wrs-wrc.names is the file containing the name of the object, so yolo will use this as the identifier. At this point, the structure of “data” will be
```
data
├── images
|    └── pulley
|           └── (image-pose pairs)
├── mesh
|    ├── MBRFA30-2-P6.obj
|    └── MBRFA30-2-P6.mtl(this is optional. If you do not want it, then delete line “mtllib” in .obj)
├── raw
|    └── pulley
|           └── (image-pose pairs)
|-processed
|      ├── pulley_refine
|      |     └── (preprocessed data)
|      ├── pulley_iterative_refine
|      |     └── (preprocessed data)
|      └── pulley_rot
|            └──  (preprocessed data)
└── wrs-wrc.names
```

3. If you want to update configuration.py, then you need to run python setup.py install. Please ensure the object name in configuration.py is consistent with the object id in wrs-wrc.names.<br/>
4. Go to this section in “src/models/testOnpic.py” change the “input-5.jpg” to the image you want to test<br/>
```
   # read image
   frame = cv2.imread("input-5.jpg")
   demo = frame.copy()
   rot_frame = frame.copy()
   refine_frame = frame.copy()
```


Then run
```
python src/models/testOnPic.py
```
For test


### Author: Jiaming Hu

## project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    |   ├── images         <- image and pose pair which are not verified yet
    |   ├── mesh           <- object mesh files
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   ├── *_rot      <- The data used for rot-classifier
    │   │   └── *_refine   <- The data used for pose refinement net
    |   ├── *.names        <- the object id which is used by yolo
    │   └── raw            <- image and pose pair which are verified yet
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── model.py       <- Rot-classifier, RefineNet
    │   └── models.py      <- Yolo model
    │
    ├── notebooks          <- Jupyter notebooks.(used for testing)
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py              <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                   <- Source code for use in this project.
    │   ├── __init__.py       <- Makes src a Python module
    │   ├── common            <- Helping functions
    │   │   ├── __init__.py
    │   │   ├── DataLoader.py <- Data loaders
    │   │   ├── object_model.py
    │   │   ├── fscore.py
    │   │   ├── iou.py
    │   │   └── chamfer2D     <- chamfer distance losss for 2d
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── getLabelFromImage.py             <- used to collect imagae and pose pair
    │   │   ├── getLabelFromImage16.py           <- used to collect imagae and pose pair
    |   |   ├── labelverifier.py                 <- used to verify the data
    |   |   ├── make_iterative_refine_dataset.py 
    │   │   ├── make_rot_dataset.py
    │   │   └── make_refine_dataset.py
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── chamfer2D
    │   │   ├── chamfer3D
    │   │   ├── detect.py                          <- API for ros function
    │   │   ├── poseUtil.py
    │   │   ├── predict_flow_model.py
    │   │   ├── precit_refine_model.py             <- RefineNet test
    |   │   ├── predict_rot_model.py               <- rot-classifier test
    │   │   ├── testOnPic.py                       <- script to test the whole pipeline on image
    │   │   ├── train_flow_model.py                <- flownet train file
    │   │   ├── train_refine_model.py              <- RefineNet train file
    │   │   ├── train_refine_model_iterative.py    <- Iterative refineNet train file
    │   │   └── train_rot_model.py                 <- rot-classifier train file
    │   │
    │   ├── utils          <- utils of yolo
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   |   └── visualize.py
    │   └── configuration.py
    ├── wandb
    ├── weights            <- all weights of models
    ├── get_pose.py        <- ros server
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

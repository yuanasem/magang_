# XarpAi Lung Opacity Detector
### An Ai powered desktop app that auto detects opacities on chest x-rays.



<br>
<img src="https://github.com/vbookshelf/XarpAi-Lung-Opacity-Detector/blob/main/images/tb0688.png" height="400"></img>
<i>Sample image<br>This opacity is an indicator of TB</i><br>
<br>


The XarpAi Lung Opacity Detector is a proof-of-concept for a free, open-source desktop app that uses artificial intelligence to detect opacities on chest x-rays. 

Opacities are characteristic signs of lung diseases like TB and Pneumonia. This app analyses chest x-rays and draws bounding boxes around opacities. Radiologists can then review these areas of interest and use their clinical judgement to make a final diagnosis.

There’s a shortage of radiologists in poor countries. In 2015, Rwanda had 11 radiologists to serve its population of 12 million people. Liberia, with a population of four million, had two practising radiologists. This app provides high volume diagnosis support. It can help overwhelmed radiologists triage x-rays and speed up their workflows.

The predictions are made by a Pytorch Faster R-CNN model. The model was fine tuned on data from four chest x-ray datasets:

- The TBX11K Tuberculosis dataset
- The Kaggle RSNA Pneumonia Detection Challenge
- The Kaggle VinBigData Chest X-ray Abnormalities Detection competition
- The Kaggle SIIM-FISABIO-RSNA COVID-19 Detection competition

Although the app outputs opacity bounding boxes, the model was also trained to detect lungs i.e. it predicts a bounding box that surrounds both lungs. If the model fails to detect the lungs then the app outputs an error message.

The model was validated on an 80/20 train test split. It was also tested on three out of sample datasets:

- The Shenzhen and Montgomery Tuberculosis datasets
- The DA and DB Tuberculosis datasets
- Child Chest X-Ray Pneumonia dataset

These out of sample datasets don’t have annotated opacity bounding boxes. Therefore, accuracy was used as a rough metric - if the target was positive (e.g. positive for TB) and the model predicted a bounding box, the model was deemed to have made a correct prediction. This validation approach is not rigorous. But it’s a quick and simple way to get a feel for the model’s capability.

Results on the 20% validation data:<br>
- map@0.5: 0.776<br>
accuracy: 0.91<br>

Accuracy on out of sample datasets:
- Shenzhen and Montgomery TB datasets: 0.84
- DA and DB TB datasets: 0.85
- Child Chest X-Ray Pneumonia dataset: 0.83

Chest x-rays can be difficult for humans to read. One study (TBX11k paper) found that radiologists have a 68.7% accuracy when diagnosing TB on chest x-rays. Using that number for context, the model’s test results look very good. The good performance on the child pneumonia data is surprising because the training data didn’t include a large number of child x-rays. 

These results show that this opacity detection app could be helpful when diagnosing lung diseases like TB and Pneumonia.

Thank you for trying this app. Please feel free to share your feedback in the discussion forum on Kaggle.
https://www.kaggle.com/datasets/vbookshelf/xarpai-lung-opacity-detector/discussion

<br>

## Demo

<br>
<img src="https://github.com/vbookshelf/XarpAi-Lung-Opacity-Detector/blob/main/images/xarpai-opacity.gif" height="400"></img>
<i>Demo showing what happens after a user submits three images</i><br>
<br>

<br>

## 1- Main Features

- Free to use.
- Can run on an ordinary computer. No need for large amounts of RAM or a GPU.
- Runs locally without needing an internet connection
- Completely transparent. All code is accessible and therefore fully auditable.
- Takes images in dicom, png or jpg format as input
- Can analyze multiple images simultaneously
- Uses the computer’s cpu. A gpu would make the app much faster, but it's not essential.
- Results are explainable because it draws bounding boxes around detected opacities
- Patient data remains private because it never leaves the user’s computer
- Images are interactive. Clicking an image causes the bounding boxes to disappear.


<br>

## 2- Cons

- It’s not a one click setup. The user needs to have a basic knowledge of how to use the command line to set up a virtual environment, download requirements and launch a python app. However, a high-schooler who knows Python could set this app up in a few minutes.
- The model’s ability to generalize to real world data is unproven.
- The model may predict multiple overlapping bounding boxes. I've not fixed this because this is a prototype and the bounding boxes can be hidden by simply clicking on the image.
- The model predicts false positives.

<br>



## 3- How to zoom into the image

To magnify the image use the desktop zoom feature that’s built into both Mac and Windows 10. 

Place the mouse pointer on the area that you want to magnify then:
- On Mac, move two fingers apart on the touchpad
- On Windows, hold down the windows key and press the + key

<br>

## 4- How to run this app

### First download the project folder from Kaggle

I've stored the project folder (named xarpai-lung-opacity-detector) in a Kaggle dataset.<br>
https://www.kaggle.com/datasets/vbookshelf/xarpai-lung-opacity-detector


I suggest that you download the project folder from Kaggle instead of from this GitHub repo. This is because the project folder on Kaggle includes the trained model. The project folder in this repo does not include the trained model because GitHub does not allow files larger than 25MB to be uploaded.<br>
The model is located inside a folder called TRAINED_MODEL_FOLDER, which is located inside the xarpai-lung-opacity-detector folder:<br>
xarpai-lung-opacity-detector/TRAINED_MODEL_FOLDER/

<br>

### System Requirements

You'll need about 1.5GB of free disk space. Other than that there are no special system requirements. This app will run on a CPU. I have an old 2014 Macbook Pro laptop with 8GB of RAM. This app runs on it without any issues.


<br>

### Overview

This is a standard flask app. The steps to set up and run the app are the same for both Mac and Windows.

1. Download the project folder.
2. Use the command line to pip install the requirements listed in the requirements.txt file. (It’s located inside the project folder.) 
3. Run the app.py file from the command line.
4. Copy the url that gets printed in the console.
5. Paste that url into your chrome browser and press Enter. The app will open in the browser.

This app is based on Flask and Pytorch, both of which are pure python. If you encounter any errors during installation you should be able to solve them quite easily. You won’t have to deal with the package dependency issues that happen when using Tensorflow.

<br>

### Detailed setup instructions

The instructions below are for a Mac. I didn't include instructions for Windows because I don't have a Windows pc and therefore, I could not test the installtion process on windows. If you’re using a Windows pc then please change the commands below to suit Windows. 

You’ll need an internet connection during the first setup. After that you’ll be able to use the app without an internet connection.

If you are a beginner you may find these resources helpful:

The Complete Guide to Python Virtual Environments!<br>
Teclado<br>
(Includes instructions for Windows)<br>
https://www.youtube.com/watch?v=KxvKCSwlUv8&t=947s

How To Create Python Virtual Environments On A Mac<br>
https://www.youtube.com/watch?v=MzuGMSw8la0&t=167s

<br>

```

1. Download the project folder, unzip it and place it on your desktop.
In this repo the project folder is named: xarpai-lung-opacity-detector
Then open your command line console.
The instructions that follow should be typed on the command line. 
There’s no need to type the $ symbol.

2. $ cd Desktop

3. $ cd project_folder

4. Create a virtual environment. (Here it’s named myvenv)
This only needs to be done once when the app is first installed.
You'll need to have python3.8 available on your computer.
When you want to run the app again you can skip this step.
$ python3.8 -m venv myvenv

5. Activate the virtual environment
$ source myvenv/bin/activate

4. Install the requirements.
This only needs to be done once when the app is first installed.
When you want to run the app again you can skip this step.
$ pip install -r requirements.txt

5. Launch the app.
This make take a few seconds the first time.
$ python app.py

6. Copy the url that gets printed out (e.g. http://127.0.0.1:5000)

7. Paste the url into your chrome browser and press Enter. The app will launch in the browser. 

8. To stop the app type ctrl C in the console.
Then deactivate the virtual environment.
$ deactivate

```

There are sample images in the sample_xray_images folder. You can use them to test the app.

While the app is analyzing, please look in the console to see if there are any errors. If there are errors, please do what’s needed to address them. Then relaunch the app.

<br>


## 5- Model Training and Validation

The model card contains a summary of the training and validation datasets as well as the validation and test results. There's also info about the app. Please refer to this document:<br>
https://github.com/vbookshelf/XarpAi-Lung-Opacity-Detector/blob/main/xarpai-lung-opacity-detector-v1.0/static/assets/Model-Card-and-App-Info-v1.0.pdf

There are many jupyter notebooks for this project. All are stored on Kaggle, so there's full traceability. Four of the main notebooks are stored in this repo in the folder called "Notebooks".<br>
https://github.com/vbookshelf/XarpAi-Lung-Opacity-Detector/tree/main/xarpai-lung-opacity-detector-v1.0/Notebooks


Exp_103<br>
The code to train the model.

Exp_104<br>
The code to test the model on the Shenzhen and Montgomery TB datasets.

Exp_105<br>
The code to locally validate the model on 20% of the data.

Exp_106<br>
The code to test the model on Child Chest X-Ray Pneumonia dataset.

Exp_107<br>
The code to test the model on the DA and DB Tuberculosis datasets.

<br>

## 6- Licenses

The app code is available under an MIT License. But please note that the trained model can't be used commercially because some of the training data is not licensed for commercial use.

<br>

## 7- Links to algorithm and datasets

#### Mask R-CNN<br>
Paper: https://arxiv.org/abs/1703.06870

<br>

I sampled data from the following datasets. I used this data to train a model to detect and isolate lungs and to train a model to detect and isolate opacities.

#### Shenzhen and Montgomery datasets
- Download from Kaggle: https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities<br>
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/


#### TBX11K Simplified
- Download from Kaggle: https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified<br>
- Paper: https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Rethinking_Computer-Aided_Tuberculosis_Diagnosis_CVPR_2020_paper.pdf


#### Belarus dataset
- Download: https://github.com/frapa/tbcnn/tree/master/belarus<br>
- Paper: https://www.nature.com/articles/s41598-019-42557-4

#### DA and DB datasets
- Download from Kaggle: https://www.kaggle.com/datasets/vbookshelf/da-and-db-tb-chest-x-ray-datasets<br>
- Download: https://sourceforge.net/projects/tbxpredict/files/data/<br>
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4229306/


#### VinDr-CXR dataset
- A version of this dataset was used for a Kaggle competition: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/overview
- Download: https://physionet.org/content/vindr-cxr/1.0.0/<br>
- Paper: https://www.nature.com/articles/s41597-022-01498-w

#### VinDr-PCXR dataset
- Download: https://physionet.org/content/vindr-pcxr/1.0.0/<br>
- Paper: https://www.medrxiv.org/content/10.1101/2022.03.04.22271937v1.full-text

#### Tuberculosis (TB) Chest X-ray Database
- Download from Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset<br>
- Paper: https://ieeexplore.ieee.org/document/9224622

#### Child Chest X-Ray Images (Version 2)
- paultimothymooney dataset version 2 on Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia<br>
- andrewmvd dataset version 2 on Kaggle: https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray<br>
- Paper: https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5


#### RSNA Pneumonia Detection Challenge
- Download from Kaggle: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data<br>
- Download from the RSNA website: https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018<br>
- Paper: https://pubs.rsna.org/doi/10.1148/ryai.2019180041

#### SIIM-FISABIO-RSNA COVID-19 Detection
- Dowload from Kaggle: https://www.kaggle.com/competitions/siim-covid19-detection/data
- Paper: https://osf.io/532ek/

<br>

I used version 1 of the VGG Image Annotator (VIA) to create a dataset of lung annotations.

#### The VIA Annotation Software for Images, Audio and Video<br>
- Paper: https://www.robots.ox.ac.uk/~vgg/software/via/docs/dutta2019vgg_arxiv.pdf
- Website: https://www.robots.ox.ac.uk/~vgg/software/via/

<br>

## 8- Acknowledgements

Many thanks to Kaggle for the free GPU and the other dataset resources they provide. I don't have my own GPU or big data cloud storage. Therefore, this project would not have been possible without Kaggle's free resources.

Thanks to Eric Chen. His blog post "Fine-tuning Mask-RCNN using PyTorch" helped demystify the Pytorch Mask-RCNN workflow for me. Using a Pytorch based workflow was key to being able to easily train and deploy this model.

Thanks to Guanshuo Xu. It was from his 10th place Covid-19 Kaggle Competition solution that I got the idea to assign a bounding box to an entire normal image. By doing this the model is able to predict a label for images without objects.

Also many thanks to all the researchers who so generously made their chest x-ray datasets publicly available.

<br>

## 9- References and Resources

Fine-tune PyTorch Pre-trained Mask-RCNN by Eric Chen<br>
https://haochen23.github.io/2020/06/fine-tune-mask-rcnn-pytorch.html#.Y-0MX-xBzUI

Beagle Detector: Fine-tune Faster-RCNN by Eric Chen<br>
https://haochen23.github.io/2020/06/fine-tune-faster-rcnn-pytorch.html#.Y-2VjexBzUI

COVID-19 AI Detection Challenge (2021) 10th place solution by Guanshuo Xu<br>
https://www.youtube.com/watch?v=voEUo5p5uXo


Kaggle COVID-19 Comp solutions<br>
https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/covid-19-al-detection-challenge-2021

VGG Image Annotator (VIA)<br>
https://www.robots.ox.ac.uk/~vgg/software/via/

List of TB and Pneumonia Chest X-ray Datasets<br>
https://github.com/vbookshelf/List-of-TB-and-Pneumonia-Datasets

Flask experiments<br>
https://github.com/vbookshelf/Flask-Experiments

Chat-GPT<br>
https://openai.com/blog/chatgpt/

How to augment boxes together with the images<br>
https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/<br>
Albumentations Github<br>
https://github.com/albumentations-team/albumentations<br>
A list of transformations that support bounding boxes<br>
https://albumentations.ai/docs/getting_started/transforms_and_targets/

<br>

## 10- Lessons learned

- Classification models can produce stellar local validation results, but they can fail miserably on out of sample x-ray data. Real world chest x-rays can come from a variety of sources and vary in quality. This is important to remember when trying to reduce the number of false positives by using a classification plus detection workflow.


- Use CHAT-GPT as a consulting radiologist. It answers medical questions clearly and concidely, in a way that ordinary people can understand. CHAT-GPT has passed the medical board exam. Try asking a question like: How do radiologists differentiate between TB and Pneumonia?

- Albumentations can augment images and change their assiciated bounding boxes. But take care when using rotations because the augmented bounding boxes are not tight and accurate.

- Some datasets may be very large. Not everyone has the resources to download and store big datasets. However, in many x-ray datasets only a small percentage of the data consists of positive samples (e.g. TB images). The vast majority are normal images. Therefore, its better to use the dataset's API to download only the positive samples and a percentage of the negative samples - one at a time. Download the images using a Kaggle notebook and store the images in the notebook's output. In this way you'll be taking advantage of Kaggle's fast internet speed and data storage capability.

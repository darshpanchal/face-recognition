## Face Recognition

Face recognition on a video using Edgetpu. 
Looks for a face from video and saves the frame where it detects.


#### Prerequisites:

- You need to setup Coral USB accelerator on your computer by installing tflite-runtime, pycoral and edgetpu-runtime. Follow this link for tutorial https://coral.ai/docs/accelerator/get-started/
- You need a trained edgetpu model(.tflite) and a label(.txt) file. You can use this tutorial https://coral.ai/docs/edgetpu/retrain-classification-ondevice/
- Haar cascade xml file required. You can get one from github of OpenCV.

#### Dependencies:

- Pycoral
- tflite-runtime==2.5.0
- PIL
- Numpy
- OpenCV
- argparse

faceextract.py is a used to extract faces from a video frame using opencv and haarcascades.

#### Usage:

> python recognize.py --model *path_to_model* --labels *path_to_label_file* \
>        --haar *path_to_haar_cascade* --video *path_to_video_file* --find *label_to_find_from_model*

### Issues:

- Low accuracy in face detection using haar cascade.

### Tasks:

- [ ] Using face detection neural network model to detect face instead of haar cascade to improve accuracy.

from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import numpy as np
import pandas as pd
import os
import cv2
import shutil
from flask import request

import torch
import torchvision
import torchvision.transforms as T


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way

    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image

    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im



# Draw the boxes on the image
def draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=20):
    """
    Set text=None to only draw a bbox without
    any text or text background.
    E.g. set text='Balloon' to write a
    title above the bbox.

    Output:
    Returns an image with one bounding box drawn.
    The title is optional.
    To draw a second bounding box pass the output image
    into this function again.

    """

    w = xmax - xmin
    h = ymax - ymin

    # Draw the bounding box
    # ......................

    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    bbox_color = (255, 255, 255)
    bbox_thickness = line_thickness

    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness)

    # Draw the background behind the text
    # ....................................

    # Only do this if text is not None.
    if text:
        # Draw the background behind the text
        text_bground_color = (0, 0, 0)  # black
        cv2.rectangle(image, (xmin, ymin - 150), (xmin + w, ymin), text_bground_color, -1)

        # Draw the text
        text_color = (255, 255, 255)  # white
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (xmin, ymin - 30)
        fontScale = 3
        thickness = 10

        image = cv2.putText(image, text, origin, font,
                            fontScale, text_color, thickness, cv2.LINE_AA)

    return image


# How to pad an image to a square
# We pad the right and the bottom so that if there are bbox coords
# then they won't be affected by the padding. That's because the origin for
# the coords is in the top left corner.

def pad_image_to_square(image):

    """
    Pads an image to a square.
    Accepts bot grayscale and multi channel images.
    """

    # Get the image shape
    shape_tuple = image.shape

    height = image.shape[0]
    width = image.shape[1]

    # Function to pad each channel of an image to a square
    # Also pads a grayscale image to a square.
    def pad_image_channel(image_channel, height, width):

        pad_amt = abs(height - width)

        if height == width:
            pad_channel = image_channel

        elif height > width:  # pad right
            top = 0
            bottom = 0
            left = 0
            right = pad_amt

            pad_channel = np.pad(image_channel, pad_width=[(top, bottom), (left, right)], mode='constant')

        else:  # if width > height then pad bottom

            top = 0
            bottom = pad_amt
            left = 0
            right = 0

            pad_channel = np.pad(image_channel, pad_width=[(top, bottom), (left, right)], mode='constant')

        return pad_channel

    # If image is grayscale i.e. shape (height, width)
    if len(shape_tuple) == 2:

        # pad the image
        padded_image = pad_image_channel(image, height, width)

    # If the image is not grayscale i.e. shape (height, width, num_channels)
    elif len(shape_tuple) == 3:

        # get the number of channels
        num_channels = image.shape[2]

        for j in range(0, num_channels):

            # select the channel
            image_channel = image[:, :, j]

            # pad the channels
            padded_channel = pad_image_channel(image_channel, height, width)

            if j == 0:
                padded_image = padded_channel

            else:
                # Stack the channels along the channel axis
                padded_image = np.dstack((padded_image, padded_channel))

    return padded_image



def process_images(image_file_list):

    # Create proc_images_dir.
    if os.path.isdir('static/proc_images_dir') == False:
        proc_images_dir = os.path.join('static', 'proc_images_dir')
        os.mkdir(proc_images_dir)


    for i, image_fname in enumerate(image_file_list):

        # Get the extension
        ext = image_fname.split('.')[1]

        # Get the path to the file
        path = 'uploads/' + image_fname

        # If the file is a dicom file
        dicom_ext_list = ['dcm', 'dicom']

        if ext in dicom_ext_list:

            # Read the dicom file
            image = read_xray(path)

            # Note that the image will be converted to grayscale in
            # the predict_on_all_images() function.

            # Pad the image to make it square.
            # Pad right and bottom so that the bbox coords
            # are not affected.
            image = pad_image_to_square(image)

            # Save the image
            new_fname = image_fname.replace(ext, 'png')
            new_path = 'static/proc_images_dir/' + new_fname

            check = cv2.imwrite(new_path, image)

        else:

            # Read the png or jpg file
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)



            # Pad the image to make it square.
            # Pad right and bottom so that the bbox coords
            # are not affected.
            image = pad_image_to_square(image)

            # Save the image
            new_path = 'static/proc_images_dir/' + image_fname

            check = cv2.imwrite(new_path, image)


def process_fasterrcnn_preds(pred, conf_threshold):

    # Get the pred scores
    pred_scores_list = list(pred[0]['scores'].detach().cpu().numpy())

    # Get the pred labels
    pred_labels_list = list(pred[0]['labels'].detach().cpu().numpy())

    # Get the pred bounding boxes, format is [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
    pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
    pred_boxes_list = [list(item) for item in pred_boxes]

    fin_scores_list = []
    fin_labels_list = []
    fin_boxes_list = []

    if len(pred_scores_list) != 0:

        for i, item in enumerate(pred_scores_list):

            if item > conf_threshold:
                fin_scores_list.append(item)
                fin_labels_list.append(pred_labels_list[i])
                fin_boxes_list.append(pred_boxes_list[i])

    pred_dict = {
        'pred_scores': fin_scores_list,
        'pred_labels': fin_labels_list,
        'pred_boxes': fin_boxes_list
    }

    # Empty lists will be returned if the model did
    # not detect any bboxes
    return pred_dict




def create_pred_dataframe(pred_dict, fname, image_height, image_width):

    num_labels = len(pred_dict['pred_labels'])

    # If the model did not predict any boxes then
    # then the pred_labels list will be empty, [].
    if num_labels == 0:

        # Create a dataframe
        # Use a number e.g. 0 and not a str.
        # A str changes the dtype of the column which
        # will cause errors later.
        empty_dict = {
            'xmin': [0],
            'ymin': [0],
            'xmax': [0],
            'ymax': [0],
            'pred_score': [0],
            'pred_labels': [2],
            'fname': fname,
            'orig_image_height': image_height,
            'orig_image_width': image_width
        }

        df = pd.DataFrame(empty_dict)

    else:

        # Create the dataframe
        df1 = pd.DataFrame(pred_dict)
        # Add a fname column
        df1['fname'] = fname

        # Create a numpy array
        boxes_np = np.array(list(df1['pred_boxes']))

        # Use the np array to create a dataframe
        cols = ['xmin', 'ymin', 'xmax', 'ymax']
        df2 = pd.DataFrame(boxes_np, columns=cols)

        # Concat side to side
        df = pd.concat([df1, df2], axis=1)
        # Remove the pred_boxes column
        df = df.drop('pred_boxes', axis=1)

        # Add the height and width columns
        df['orig_image_height'] = image_height
        df['orig_image_width'] = image_width

    # If the label is 2 i.e. normal (no opacity) then
    # set all coords for that row to 0.
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    for i in range(0, len(df1)):

        pred_label = df.loc[i, 'pred_labels']
        xmin = df.loc[i, 'xmin']
        ymin = df.loc[i, 'ymin']
        xmax = df.loc[i, 'xmax']
        ymax = df.loc[i, 'ymax']

        if pred_label == 2:

            xmin_list.append(0)
            ymin_list.append(0)
            xmax_list.append(0)
            ymax_list.append(0)

        else:
            xmin_list.append(xmin)
            ymin_list.append(ymin)
            xmax_list.append(xmax)
            ymax_list.append(ymax)

    df['xmin'] = xmin_list
    df['ymin'] = ymin_list
    df['xmax'] = xmax_list
    df['ymax'] = ymax_list

    return df


def predict_on_all_images(model_list, device, image_list, threshold):

    # Create pred_images_dir.
    if os.path.isdir('static/pred_images_dir') == False:
        pred_images_dir = 'static/pred_images_dir'
        os.mkdir(pred_images_dir)

    print('Starting prediction...')
    print(os.listdir('static/proc_images_dir'))

    model_path_0 = f"TRAINED_MODEL_FOLDER/{model_list[0]}"

    # Load the trained model
    model = torch.load(model_path_0, map_location=torch.device('cpu'))
    # model = torch.load(path_model)

    # Put the model in eval model
    model.eval()

    # Send the model to the device
    model.to(device)

    num_preds_dict = {}

    for i, fname in enumerate(image_list):

        print(f'Predicting on image {i+1} of {len(image_list)}...')

        # Keep the original fname that may have a dicom extension.
        # We need to display this fname on the page.
        orig_fname = fname

        # If the file has a dicom fname them
        # replace the extension with png.
        ext = fname.split('.')[1]
        dicom_ext_list = ['dcm', 'dicom']

        if ext in  dicom_ext_list:
            fname = fname.replace(ext, 'png')


        path = 'static/proc_images_dir/' + fname

        # Load the image with PIL
        # Don't convert to grayscale
        #image = Image.open(path)

        # Load the image with PIL
        # Load and convert the image to Grayscale
        image = Image.open(path).convert("L")

        # Pad the image to a square.
        # Note that the images were already padded to a square in then process_images() function

        # Get the sizes of the image

        # Convert the PIL image into a numpy array
        image1 = np.array(image)
        # Get the height and width
        image_height = image1.shape[0]
        image_width = image1.shape[1]

        # Transform the image to a torch tensor.
        my_transform = T.Compose([T.ToTensor()])
        image = my_transform(image)

        # Send the image to the device
        image = image.to(device)

        # Predict on the image
        pred = model([image])

        pred_dict = process_fasterrcnn_preds(pred, threshold)


        # Create pred dataframe
        df1 = create_pred_dataframe(pred_dict, fname, image_height, image_width)

        # Concat the dataframe of each image
        if i == 0:
            df_fin = df1
        else:
            df_fin = pd.concat([df_fin, df1], axis=0)



        # Draw the bbox on the image
        # Note: Here we need to not draw the bboxes on normal images.
        # ---------------------------

        # Load the original image
        path = 'static/proc_images_dir/' + fname
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)


        pred_boxes_list = pred_dict['pred_boxes']
        pred_labels_list = pred_dict['pred_labels']

        # 2 is the label for normal images.
        # When the image is normal the model predicts a bbox
        # that covers the whole image.
        # Here we will ensure that the bbox for a pred of 2 does not get
        # drawn on the image or included in the num_preds count.

        new_label_list = []
        new_bbox_list = []

        # This is a list of labels for which we don't want to draw bboxes
        # on the image.
        # Pred label 2 refers to a bbox around a normal image
        # Pred label 3 refers to a bbox around the lungs
        # Remove label 3 from this list if you want to see the
        # bbox drawn around the lungs. But remember that the code will
        # include this lung bbox in the count of the number of opacities detected.
        ignore_labels_list = [2, 3] #[2, 3]

        lungs_detected = 'no'

        for j in range(0, len(pred_labels_list)):

            label = pred_labels_list[j]
            bbox = pred_boxes_list[j]

            # Check that the the lungs were detected
            if label == 3:
                lungs_detected = 'yes'


            if label not in ignore_labels_list:
                new_label_list.append(label)
                new_bbox_list.append(bbox)

        num_preds = len(new_label_list)


        # Add a key value pair to the dict
        #num_preds_dict[orig_fname] = num_preds
        num_preds_dict[orig_fname] = (num_preds, lungs_detected)

        for i in range(0, len(new_bbox_list)):

            coords_list = new_bbox_list[i]

            #image = draw_pred_bbox_on_image(image, bbox_coords)

            xmin = int(coords_list[0])
            ymin = int(coords_list[1])
            xmax = int(coords_list[2])
            ymax = int(coords_list[3])

            image = draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=2)


        # Save the image with the bboxes drawn in
        dst = os.path.join('static/pred_images_dir/', fname)
        cv2.imwrite(dst, image)

    # Save df_fin as a csv file.
    # We can use this csv file later to control
    # what happens when the user clicks on the image.
    path = 'df_fin_preds.csv'
    df_fin.to_csv(path, index=False)

    print('Prediction completed.')

    # Sort the order that the preds will be displayed.
    # Images that have an opacity should be shown first.

    list1 = []
    list2 = []

    for key, value in num_preds_dict.items():

        # the value is a tuple: (num_preds, lungs_detected)
        if value[0] > 0:
            list1.append(key)
        else:
            list2.append(key)

    # Join the python lists
    sorted_image_list = list1 + list2

    assert len(sorted_image_list) == len(image_list)

    return num_preds_dict, sorted_image_list




def delete_user_submitted_data():

    """
    Note:
    This function does not delete the images in 'static/pred_images_dir'.
    The app needs the png images in this folder to display them on the main page.
    The 'static/pred_images_dir' folder gets deleted each time the user submits new files.

    """
    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('uploads') == True:
        shutil.rmtree('uploads')

    # Delete folders and their contents.
    # This is for data security.
    #if os.path.isdir('static/proc_images_dir') == True:
        #shutil.rmtree('static/proc_images_dir')


    # Delete folders and their contents.
    # This is for data security.
    #if os.path.isdir('static/pred_images_dir') == True:
        #shutil.rmtree('static/pred_images_dir')



# When the user clicks on the image
# this function makes the bboxes disappear.
def hide_all_bboxes(image_fname):

    # Delete the analysis images folder if it exists.
    if os.path.isdir('static/analysis_images_dir') == True:
        shutil.rmtree('static/analysis_images_dir')
        print('Folder deleted.')

    # Create analysis_images_dir.
    if os.path.isdir('static/analysis_images_dir') == False:
        analysis_images_dir = 'static/analysis_images_dir'
        os.mkdir(analysis_images_dir)


    # Load a fresh image without dots or bboxes
    path = os.path.join('static/proc_images_dir', image_fname)
    image = cv2.imread(path)

    # Change the image_fname to indicate that the bboxes have been hidden
    image_fname = 'no_bboxes_' + image_fname


    # We will use this to create a new folder name.
    k = str(99)

    # The problem:
    # We want to display the same image each time with just the bbox drawn in a different place.
    # But in the new_image_str code below the browser will not change the displayed image
    # if the src path to the new image is the same as for the previous image.
    # Solution:
    # We will store each changed image in a different folder. This will change the src path while
    # still keeping the same image_fname. We need the fname to stay the same because each time
    # we need to load the image that the user originally submitted, which is stored in png_images_dir.

    # We can change the folder name each time because
    # we only need the file name that's at the end of the src attribute.
    # As long as the file name stays the same each time everything will work.
    new_image_str = f"""<img id="selected-image" onclick="get_click_coords(event, this.src)"  class="w3-round unblock" src="/static/analysis_images_dir/{k}/{image_fname}"  height="580" alt="Wheat">"""


    # Only if the user clicked inside a bbox
    if new_image_str != 'None':

        print('User clicked on the image.')

        # Create analysis_images_dir.
        if os.path.isdir(f'static/analysis_images_dir/{k}') == False:
            analysis_images_dir = f'static/analysis_images_dir/{k}'
            os.mkdir(analysis_images_dir)

        # save the image
        dst = os.path.join(f'static/analysis_images_dir/{k}', image_fname)
        cv2.imwrite(dst, image)


    # If the user did not click inside a bbox then
    # new_image_str == 'None'.
    # Then the javascript code won't change the image on the page.
    # The existing image will remain as is.
    output = {
                'new_image_str': new_image_str
              }

    return output



# When the user clicks on an image
# this function makes the bboxes appear.
def show_all_bboxes(image_fname):

    # Remove the name 'no_bboxes_' from the fname, if it's there
    image_fname = image_fname.replace('no_bboxes_', '')

    # Load a fresh image without dots or bboxes
    #path = os.path.join('static/pred_images_dir', image_fname)
    #image = cv2.imread(path)

    # Load the predicted image that has the bboxes already drawn in
    new_image_str = f"""<img id="selected-image" onclick="get_click_coords(event, this.src)"  class="w3-round unblock" src="/static/pred_images_dir/{image_fname}"  height="580" alt="chest x-ray">"""


    output = {
                'new_image_str': new_image_str
              }

    return output
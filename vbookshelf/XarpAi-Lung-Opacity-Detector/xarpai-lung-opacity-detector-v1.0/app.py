
"""
NOTES

1- Yolo version being used: Yolov5-5.0
The yolov5-5.0 folder has been named yolov5.

2- Model version being used: exp29_best.pt

3- The trained model has been placed here:
yolov5/TRAINED_MODEL_FOLDER/

4- In this app the endpoints send and receive data using Ajax.
This allows the web pages to be updated without being refreshed each time.
This is a simple example that shows how the process works:
https://github.com/vbookshelf/Flask-Experiments/tree/main/Exp_11-working-ajax-flask-request-response-example-template

5- I changed line 173 in detect.py (located inside the yolov5 folder) to include 'PyYAML' in the list
of packages that Yolo should not check for.
This fixed an error in PyCharm.
Line 73: check_requirements(exclude=('pycocotools', 'thop', 'PyYAML'))

"""

from utils import *
import os
from flask import Flask, render_template, url_for, request, redirect, jsonify
from werkzeug.utils import secure_filename

# Specify the model name/s here.
MODEL_LIST = ['exp114_model.pt']

# 0 or 0,1,2,3 or cpu
DEVICE = 'cpu'

# Set the pred confidence threshold
THRESHOLD = 0.5


# Create an instance of the Flask class
app = Flask(__name__, static_url_path='/static')

# Note that there is code below that converts all file extensions to lower case.
app.config['ALLOWED_EXTENSIONS'] = ['.dicom', '.dcm', '.png', '.jpg', '.jpeg']

# Get the absolute path to the folder called 'static'.
# We must get this path before we change the working directory.
ABS_PATH_TO_STATIC = os.path.abspath("static")


# This endpoint loads the index.html page.
@app.route('/')
def home_func():
    return render_template('index.html')


# This is the endpoint that loads the page that displays the model card.
@app.route('/about')
def about_func():
    return render_template('more-info.html')


# This is the endpoint that loads the page that displays the FAQ.
@app.route('/faq')
def faq_func():
    return render_template('faq.html')


# This is the endpoint that loads the page that displays the FAQ.
@app.route('/feedback')
def feedback_func():
    return render_template('feedback.html')


# This endpoint contains the code for:
# - dicom file uploading
# - image processing
# - inference
# - displaying the inference results on the page
@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():

    # This try except block handles the condition where
    # a single file is uploaded and it's not a dicom file.
    # If the user uploads a mix of dicom files and non dicom files,
    # the code simply ignores the non dicom files.
    #try:

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('uploads') == True:
        shutil.rmtree('uploads')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('static/proc_images_dir') == True:
        shutil.rmtree('static/proc_images_dir')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('static/pred_images_dir') == True:
        shutil.rmtree('static/pred_images_dir')



    # Create a new folder to store uploaded image files
    if os.path.isdir('uploads') == False:
        uploads = 'uploads'
        os.mkdir(uploads)



    # Get a list.
    # my_files is the name that is used in the html code.
    file_list = request.files.getlist('my_files')

    print(file_list)

    for item in file_list:
        # Get the file name.
        # We need to secure and clean up the file name.
        fname = item.filename

        # Get the file extension e.g. .dicom
        # Convert to lower case
        extension = os.path.splitext(item.filename)[1].lower()

        # Only save the file if it has a .dicom extension
        if extension in app.config['ALLOWED_EXTENSIONS']:

            # Create a secure file name.
            # Replace any spaces with underscores.
            # Any other malicious symbols get removed.
            fname = secure_filename(fname)

            # Save the file to a folder called uploads only if it has a .dicom extension
            # We need to create a folder called uploads.
            item.save(f'uploads/{fname}')


    # Get a list of files in the uploads folder
    upfile_list = os.listdir('uploads')


    # Get only list items that are image files.
    # Sometimes there are strange hidden files in local folders

    image_list = []

    for item in upfile_list:

        # Get the file extension and convert to lower case.
        file_ext = item.split('.')[1].lower()
        file_ext_with_dot = f'.{file_ext}'

        if file_ext_with_dot in app.config['ALLOWED_EXTENSIONS']:
            image_list.append(item)



    # MAIN FUNCTIONS
    # --------------

    # Process the images and store in proc_images_dir
    # - Convert dicom images to png
    # - Make all images square
    process_images(image_list)

    # Predict on all images in proc_images_dir
    # - Make the prediction
    # - Draw the bbox on the image
    # - Store the image in pred_images_dir
    # - Created a sorted list where the images with opacities appear first
    num_preds_dict, sorted_image_list = predict_on_all_images(MODEL_LIST, DEVICE, image_list, THRESHOLD)

    print('Done')


    # ---------------


    # Create html for the images that will be loaded into the hidden image elements.
    # This will cache the images and make them instantly available when a user clicks on a link
    # to display an image.
    for i, item in enumerate(sorted_image_list):

        # If the file has a dicom fname them
        # replace the extension with png.
        ext = fname.split('.')[1]
        dicom_ext_list = ['dcm', 'dicom']

        if ext in dicom_ext_list:
            fname = fname.replace(ext, 'png')


        if i == 0:
            image_fin_str = f"""<img class="w3-round unblock" src="/static/pred_images_dir/{item}"  height="580">"""
        else:
            image_fin_str = image_fin_str + f"""<img  class="w3-round unblock" src="/static/pred_images_dir/{item}"  height="580">"""


    # Create the html for the clickable links to images.
    start_str = "<ul>"
    for i, item in enumerate(sorted_image_list):

        # Get the pred_info tuple: (num_preds, lungs_detected)
        pred_info_tuple = num_preds_dict[item]
        lungs_detected = pred_info_tuple[1]


        # Create the string that shows the number of bboxes for a given fname.
        # The dict key is the item and the value is num preds.
        #num_preds = num_preds_dict[item]
        num_preds = pred_info_tuple[0]

        if lungs_detected == 'yes':

            if num_preds == 1:
                num_str = str(num_preds) + ' ' + 'opacity detected'
            else:
                num_str = str(num_preds) + ' ' + 'opacities detected'

        elif lungs_detected == 'no':
            num_str = 'Error. Image is not a chest x-ray'


        # Here we are creating this: <ul> <li>fname1</li> <li>fname2</li> ...
        if i == 0:
            fin_str = start_str + f'<li class="row w3-text-black w3-border-right w3-border-black w3-padding-bottom" onclick="ajaxGetFilename(this.innerHTML)"><a href="#">{num_str}<br>{item}</a></li>'
        else:
            fin_str = fin_str + f'<li class="row w3-padding-bottom" onclick="ajaxGetFilename(this.innerHTML)"><a href="#">{num_str}<br>{item}</a></li>'





    # The closing </ul> is included here.
    html_str = fin_str + '</ul>' + """<script>jQuery('li').click(function(event){
                //remove all pre-existing active classes
                jQuery('.row').removeClass('w3-text-black w3-border-right w3-border-black');
        
                //add the active class to the link we clicked
                jQuery(this).addClass('w3-text-black w3-border-right w3-border-black');
                event.preventDefault();
                 });</script>"""

    # We want the first image to be displayed as the main image.
    first_fname = sorted_image_list[0]

    # If the file has a dicom fname them
    # replace the extension with png.
    ext = first_fname.split('.')[1]
    dicom_ext_list = ['dcm', 'dicom']

    if ext in dicom_ext_list:
        first_fname = fname.replace(ext, 'png')

    #main_image_str = f"""<img id="selected-image"  class="w3-round unblock" src="/static/pred_images_dir/{first_fname}"  height="580" alt="chest x-ray">"""

    # This str contains the onclick function that activates
    # When the user clicks on an image.
    main_image_str = f"""<img id="selected-image"  onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{first_fname}"  height="580" alt="chest x-ray">"""

    output_reponse = {"html_str": html_str, "main_image_str": main_image_str, "image_fin_str": image_fin_str}

    # Delete the data the user has submitted.
    # The png images in static/pred_image_dir don't get deleted here because
    # the app needs to display these images on the main page.
    # The static/pred_image_dir folder gets deleted each time the user submits new files.
    delete_user_submitted_data()

    return jsonify(output_reponse)

    """

    except:

        output_response = { "html_str": '<p>Error. Please look at the console for more info.<br>Please submit only dicom files.<br> Allowed extensions: .dicom, .dcm, .DICOM, .DCM</p>'}

        # Delete the data the user has submitted.
        # The png images in static/pred_image_dir don't get deleted here because
        # the app needs to display these images on the main page.
        # The static/pred_image_dir folder gets deleted each time the user submits new files.
        delete_user_submitted_data()

        return jsonify(output_response)

    """


# When the user clicks a file name,
# display that image as the main image on the page.
@app.route('/process_ajax', methods=['POST'])
def process_ajax():

    # Get the value of the 'file_name' key
    # Example fname: <a href="#">1 result<br>82764af4-85dc-4512-bdfe-d9e9f66e0fc4.dcm</a>
    fname = request.form.get('file_name')

    print(fname)

    # Remove the first part of the str to get this fname format:
    # 47c8858666bcce92bcbd57974b5ce522.dicom</a>
    fname = fname.split('<br>')[1]
    fname = fname.replace('</a>', '')
    ext = fname.split('.')[1]

    #print(fname)
    dicom_ext_list = ['dcm', 'dicom']
    if ext in dicom_ext_list:
        # Replace .dicom with .png
        # 47c8858666bcce92bcbd57974b5ce522.png
        image_fname = fname.split('.')[0] + '.png'

    else:
        image_fname = fname

    print(image_fname)

    # Create html code containing the image info
    info_in_html = f"""<img id="selected-image"  onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{image_fname}"  height="580" alt="chest x-ray">"""

    output = {"output1":  info_in_html}

    return jsonify(output)



# When the user clicks a file name,
# display that image as the main image on the page.
@app.route('/process_sample_ajax', methods=['POST'])
def process_sample_ajax():

    # Get the value of the 'file_name' key
    # Example fname: 0 results<br>47c8858666bcce92bcbd57974b5ce522.dicom
    fname = request.form.get('file_name')

    # Remove the first part of the str to get this fname format:
    # 47c8858666bcce92bcbd57974b5ce522.dicom</a>
    fname = fname.split('<br>')[1]
    fname = fname.replace('</a>', '')
    ext = fname.split('.')[1]


    # print(fname)
    dicom_ext_list = ['dcm', 'dicom']
    if ext in dicom_ext_list:
        # Replace .dicom with .png
        # 47c8858666bcce92bcbd57974b5ce522.png
        image_fname = fname.split('.')[0] + '.png'

    else:
        image_fname = fname

    # Create html code containing the image info
    info_in_html = f"""<img id="selected-image"  onclick="hide_show_bboxes(this.src, this.id)" class="w3-round unblock" src="/static/sample_images/{image_fname}"  height="580" alt="chest x-ray">"""

    output = {"output1":  info_in_html}

    return jsonify(output)



# When the user clicks on a predicted image
# this endpoint hides or shows the bboxes.
@app.route('/process_click_info', methods=['POST'])
def process_click_info():

    # Get the image path
    fname = request.form.get('fname')

    # Get the last item in the list
    # Example: [http://127.0.0.1:5000/static/pred_images_dir/f23b18fec1.png]
    image_fname = fname.split('/')[-1:][0]

    # When we hide all bboxes on an image we add 'no_bboxes_' to the file name.
    if 'no_bboxes_' in image_fname:

        # Load the image with the bboxes already drawn in
        output = show_all_bboxes(image_fname)

    else:
        # Remove all bboxes from the image
        output = hide_all_bboxes(image_fname)

    #return ("", 204)
    return jsonify(output)




# When the user clicks on a sample image
# this endpoint hides or shows the bboxes.
@app.route('/process_sample_image_click', methods=['POST'])
def process_sample_image_click():

    # Get the image path
    fname = request.form.get('fname')

    # Get the html element id
    id = request.form.get('id')

    # Get the last item in the list
    # Example: [http://127.0.0.1:5000/static/pred_images_dir/f23b18fec1.png]
    image_fname = fname.split('/')[-1:][0]

    # Sample images that don't have bboxes have 'noboxes_' in their file name.
    if 'noboxes_' in image_fname:

        # Remove 'no_bboxes_' from the fname
        new_fname = fname.replace('noboxes_', '')

        # Load the image with the bboxes already drawn in
        output = {
                'new_fname': new_fname,
                'id': id
              }

    # If the current image is not showing bboxes
    else:

        # Split the path into a list
        item_list = fname.split('/')

        # Add 'noboxes_' to the image_fname
        image_fname = 'noboxes_' + image_fname

        # Change the last item in the list
        last_index = len(item_list) - 1
        item_list[last_index] = image_fname

        # Join all items in the list to form the image path.
        new_fname = '/'.join(item_list)

        # Load the image with no bboxes
        output = output = {
                'new_fname': new_fname,
                'id': id
              }

    #return ("", 204)
    return jsonify(output)



# This endpoint is used to test that the app is working
@app.route('/test')
def test():
    return 'This is a test...'




if __name__ == '__main__':
    app.run()
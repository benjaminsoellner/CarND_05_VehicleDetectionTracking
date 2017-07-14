"""
Contains all model functions to perform vehicle detection and its sub-steps
"""

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import deque
import numpy as np
import matplotlib.image as mpimg
import cv2
import glob
import os

def convert_color(image, color_space):
    """
    convert an image from one color space to another
    :param image: the image as a x*y*3 numpy-array
    :param color_space: the color space, either of 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb' 
    :return: the converted x*y*3 dimensional numpy-array image
    """
    out_image = None
    if color_space != 'RGB':
        if color_space == 'HSV':
            out_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            out_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            out_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            out_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            out_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    else:
        out_image = np.copy(image)
    return out_image

def get_templates(templates_path_pattern):
    """
    Helper function to search at a pattern of paths for image files and sort them into non-car and car pictures 
    based on whether the path contains the string 'non-vehicle'
    :param templates_path: a pattern of paths to search for car and non-car images
    :return: a tuple containing a list of car image paths and a list of non-car image paths
    """
    templates_paths = glob.glob(templates_path_pattern)
    cars = []
    notcars = []
    for template_path in templates_paths:
        if 'non-vehicles' in template_path:
            notcars.append(template_path)
        else:
            cars.append(template_path)
    return cars, notcars

def draw_boxes(image, bboxes, color=(0., 0., 1.0), thick=6):
    """
    Adds bounding boxes to an image
    (from Udacity CarND 05 / lesson 05. Manual Vehicle Detection)
    :param image: x*y*3 numpy array representation of image
    :param bboxes: list of bounding boxes in the form ((x0,y0), (x1,y1))
    :param color: color in the form of (r,g,b) with all values between 0.0 and 1.0
    :param thick: thickness of bounding box drawn
    :return: the image with bounding boxes drawn
    """
    # make a copy of the image
    draw_img = np.copy(image)
    # draw each bounding box on your image copy using cv2.rectangle()
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img

def find_matches(image, template_paths):
    """
    Takes a list of template paths and matches the templates on the image, returns the bounding boxes
    (from Udacity CarND 05 / lesson 09. Template Matching)
    :param image: x*y*3 numpy array representation of image
    :param template_paths: the list of template files to scan the image for
    :return: 
    """
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    method = cv2.TM_CCOEFF_NORMED
    # Iterate through template list
    for template in template_paths:
        # Read in templates one by one
        template_image = mpimg.imread(template).astype(np.float32)
        # Use cv2.matchTemplate() to search the image
        result = cv2.matchTemplate(image.astype(np.float32), template_image, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Determine a bounding boex for the match
        width, height = (template_image.shape[1], template_image.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        # Return the list of bounding boxes
    return bbox_list

def get_hist_features(image, nbins=32, bins_range=(0., 1.)):
    """
    Gets the histogram features for an image on all 3 channels.
    (from CarND 05 / lesson 12. Histograms of Color)
    :param image: x*y*3 numpy array representation of image
    :param nbins: number of histogram bins to divide the channel features in
    :param bins_range: range (min, max) of the histogram
    :return: 5-tuple: first three values contain the histograms returned from np.histogram, 4th value contains the
      histogram center and 5th value contains all 3 histograms concatenated
    """
    # Compute the histogram of the RGB channels separately
    r = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    g = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    b = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = r[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((r[0], g[0], b[0]))
    # Return the individual histograms, bin_centers and feature vector
    return r, g, b, bin_centers, hist_features

def get_spatial_features(image, size=(32, 32)):
    """
    Gets the spatial features ofan image on all 3 channels 
    (from CarND 05 / lesson 16. Spatial Binning of Color)
    :param image: x*y*3 numpy array representation of image
    :param color_space: the color space to get the spatial features from
    :param size: possibility to resize / subsample the spatial feature space, provide a value (width, height)
    :return: returns a 1-dimensional feature vector
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(image, size).ravel()
    # Return the feature vector
    return features

def get_hog_features(image, orient, pix_per_cell, cell_per_block, transform_sqrt=False, vis=False, feature_vec=True):
    """
    Gets the hog features of an image
    (from CarND 05 / lesson 20. sci-kit Image HOG)
    :param image: x*y*3 numpy array representation of image
    :param orient: number of orientations to compute the hog features for
    :param pix_per_cell: number of pixels collected in one hog cell
    :param cell_per_block: number of cells per block, i.e. defining the overlap of cells
    :param transform_sqrt: use transform_sqrt=True in skimage.hog(...) function (True/False)
    :param vis: return a visualization image (True/False) 
    :param feature_vec: return feature vector or hog cell tensor (skimage.hog(... feature_vector=...), True/False)
    :return: the hog features and, if vis was set to True, a visualization of the hog cells
    """
    if vis == True:
        features, hog_image = hog(image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_sqrt,
                                  visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_sqrt,
                       visualise=False, feature_vector=feature_vec)
        return features

def get_features_images(image_paths, hyperparams):
    """
    Get all the features of a list of images
    (see CarND 05 / lesson 22. Combine and Normalize Features & lesson 29. HOG Classify)
    :param image_paths: a list of paths to images readable by mpimg.imread(...)
    :param hyperparams: hyperparams specification, see hyperparams.py
    :return: a list of feature vectors for all images
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in image_paths:
        # Read in each one by one
        image = mpimg.imread(file)
        # Get features
        f = get_features_image(image, hyperparams)
        features.append(f)
    # Return list of feature vectors
    return features

def get_features_image(image, hyperparams):
    """
    Extract features from one single image
    (see CarND 05 / lesson 34. Search and Classify)     
    :param image: x*y*3 numpy array representation of image
    :param hyperparams: hyperparams specification, see hyperparams.py
    :return: 
    """
    # Get all relevant hyperparameters
    color_space = hyperparams['COLOR_SPACE']
    spatial_feat = hyperparams['SPATIAL_FEAT']
    spatial_size = hyperparams['SPATIAL_SIZE']
    hist_feat = hyperparams['HIST_FEAT']
    hist_bin = hyperparams['HIST_BIN']
    hist_range = hyperparams['HIST_RANGE']
    hog_orient = hyperparams['HOG_ORIENT']
    hog_cell_per_block = hyperparams['HOG_CELL_PER_BLOCK']
    hog_pix_per_cell = hyperparams['HOG_PIX_PER_CELL']
    hog_feat = hyperparams['HOG_FEAT']
    hog_channel = hyperparams['HOG_CHANNEL']
    hog_sqrt = hyperparams['HOG_SQRT']
    # Define an empty list to receive features
    img_features = []
    # apply color conversion if other than 'RGB'
    feature_image = convert_color(image, color_space)
    # Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = get_spatial_features(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    # Compute histogram features if flag is set
    if hist_feat:
        _, _, _, _, hist_features = get_hist_features(feature_image, nbins=hist_bin, bins_range=hist_range)
        img_features.append(hist_features)
    # Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], hog_orient, hog_pix_per_cell,
                        hog_cell_per_block, vis=False, transform_sqrt=hog_sqrt, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], hog_orient,
                    hog_pix_per_cell, hog_cell_per_block, vis=False, transform_sqrt=hog_sqrt, feature_vec=True)
        img_features.append(hog_features)
    # Return concatenated array of features
    return np.concatenate(img_features)

def generate_classifier(templates_path_pattern, hyperparams):
    """
    Generate car/noncar image classifier pipeline from car/noncar templates
    :param templates_path_pattern: the filename pattern where to get the car/noncar images from
    :param hyperparams: hyperparams specification, see hyperparams.py
    :return: A 3-tuple consisting of the sklearn-classifier pipeline and the feature and label vector for the
      test set used
    """
    # Get all paths
    cars, notcars = get_templates(templates_path_pattern)
    # Extract car & non-car features
    car_features = get_features_images(cars, hyperparams)
    notcar_features = get_features_images(notcars, hyperparams)
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    # Use a linear SVC
    X_scaler = StandardScaler()
    svc = LinearSVC()
    clf = Pipeline(steps=[('StandardScaler', X_scaler), ('LinearSVC', svc)])
    clf.fit(X_train, y_train)
    # Check the prediction time for a single sample
    return clf, X_test, y_test

def persist_classifier(clf, X_test, y_test, pickle_file):
    """
    Dump a classification pipeline, as well as the features and labels used for train- and test set into a 
    pickle file 
    :param clf: the classification pipeline
    :param X_test: the feature vector of the test set
    :param y_test: the label vector of the test set
    :param pickle_file: filename of the pickle file to dump the 
    """
    joblib.dump((clf, X_test, y_test), pickle_file)

def restore_classifier(pickle_file):
    """
    Try to restore the cars/notcars classifier from a file 
    :param pickle_file: pickle filename to restore the classifier, X_test and y_test from 
    :return: None, if the file doesn't exist - otherwise: classifier (clf), test features (X_test), test labels (y_test)
    """
    if os.path.isfile(pickle_file):
        n_tuple = joblib.load(pickle_file)
        clf = n_tuple[0]
        X_test = n_tuple[1]
        y_test = n_tuple[2]
        return clf, X_test, y_test
    else:
        return None

def restore_or_generate_classifier(pickle_file, templates_path_pattern, hyperparams):
    """
    Try to restore the cars/notcars classifier from a file, if not present, generate the classifier from scratch.
    :param pickle_file: pickle filename to restore the classifier, X_test and y_test from
    :param templates_path_pattern: the filename pattern where to get the car/noncar images from
    :param hyperparams: hyperparams specification, see hyperparams.py
    :return: A 3-tuple consisting of the sklearn-classifier pipeline and the feature and label vector for the
      test set used
    """
    if pickle_file is not None:
        restored = restore_classifier(pickle_file)
        if restored is not None:
            return restored
    return generate_classifier(templates_path_pattern, hyperparams)

def slide_window(image, x_start=0, x_stop=None, y_start=0, y_stop=None, xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    takes an image, start and stop positions in both x and y, window size (x and y dimensions), and overlap fraction 
    (for both x and y) and generates a list of slide windows
    (from CarND 05 / lesson 32. Sliding Window Implementation)
    :param image: image to generate sliding windows for (needed to establish x- and y-size) 
    :param x_start: region of interest: x start position
    :param x_stop: region of interest: x stop position
    :param y_start: region of interest: y start position
    :param y_stop: region of interest: y stop position
    :param xy_window: window size in form (width,height)
    :param xy_overlap: overlap fraction in (x,y) direction
    :return: a list of windows, each in the form ((startx, starty), (endx, endy))
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_stop == None:
        x_stop = image.shape[1]
    if y_stop == None:
        y_stop = image.shape[0]
    # Compute the span of the region to be searched
    xspan = x_stop - x_start
    yspan = y_stop - y_start
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(image, windows, clf, hyperparams):
    """
    Search all windows of an image with a classifier
    (from CarND 05 / lesson 34. Search and Classify)
    :param image: image to search
    :param windows: list of sliding windows, each in the form ((x0,y0), (x1,y1))
    :param clf: classifier pipeline from sklearn
    :param hyperparams: hyperparams specification, see hyperparams.py
    :return: the windows where a match was found, each in the form ((x0,y0), (x1,y1)) + a list of confidences for each
      window
    """
    # Create an empty list to receive positive detection windows
    on_windows = []
    confidences = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # Extract features for that window using single_img_features()
        features = get_features_image(test_img, hyperparams)
        # Scale extracted features to be fed to classifier
        test_features = np.array(features).reshape(1, -1)
        # Predict using your classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
            confidences.append(clf.decision_function(test_features))
    # Return windows for positive detections
    return on_windows, confidences

def scan_single_win_size(image, clf, rescale, hyperparams, box_color=(0,0,1.0), heatmap=None):
    """
    Scans all sliding windows of one single scaling factors, does feature extraction and 
    classification, however calculates the hog features for images only once
    :param image: image to do feature sliding window classification on  
    :param clf: sklearn classification pipeline
    :param rescale: scaling factor of the window size (window size will be extracted from hyperparams)
    :param hyperparams: hyperparams specification, see hyperparams.py
    :param box_color: color to use when drawing bounding boxes on classifier hits
    :param heatmap: heatmap to use as a starting point, allowing summed-up heatmaps - heatmap will be modified
    :return: returns two values: (1) the visualized classifier matches, (2) the updated heatmap 
    """
    # get important hyperparams
    y_start = hyperparams['Y_START']
    y_stop = hyperparams['Y_STOP']
    color_space = hyperparams['COLOR_SPACE']
    spatial_feat = hyperparams['SPATIAL_FEAT']
    spatial_size = hyperparams['SPATIAL_SIZE']
    hist_feat = hyperparams['HIST_FEAT']
    hist_bin = hyperparams['HIST_BIN']
    hist_range = hyperparams['HIST_RANGE']
    hog_orient = hyperparams['HOG_ORIENT']
    hog_cell_per_block = hyperparams['HOG_CELL_PER_BLOCK']
    hog_pix_per_cell = hyperparams['HOG_PIX_PER_CELL']
    hog_feat = hyperparams['HOG_FEAT']
    hog_channel = hyperparams['HOG_CHANNEL']
    # if heatmap was not provided, initialize it
    if heatmap is None:
        heatmap = np.zeros(image.shape[0:2])
    # copy image for visualization purposes
    draw_image = np.zeros_like(image)
    # image to be analyzed is color-converted to target color space and resized to match
    # appropriately rescaled window size
    rescaled_image = convert_color(image[y_start:y_stop, :, :], color_space)
    if rescale != 1:
        imshape = image.shape
        rescaled_image = cv2.resize(rescaled_image, (np.int(imshape[1] / rescale), np.int(imshape[0] / rescale)))
    # Define number of blocks
    n_x_blocks = (rescaled_image.shape[1] // hog_pix_per_cell) - hog_cell_per_block + 1
    n_y_blocks = (rescaled_image.shape[0] // hog_pix_per_cell) - hog_cell_per_block + 1
    window = 64 # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    blocks_per_window = (window // hog_pix_per_cell) - hog_cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    n_x_steps = (n_x_blocks - blocks_per_window) // cells_per_step
    n_y_steps = (n_y_blocks - blocks_per_window) // cells_per_step
    # Prepare the hog features
    hogs = []
    if hog_feat:
        for channel in range(0, rescaled_image.shape[2]):
            hogs.append(get_hog_features(rescaled_image[:, :, channel],
                        hog_orient, hog_pix_per_cell, hog_cell_per_block, feature_vec=False))
    # Go through blocks of image step by step
    for x_window in range(n_x_steps):
        for y_window in range(n_y_steps):
            y_pos = y_window * cells_per_step
            x_pos = x_window * cells_per_step
            x_left = x_pos * hog_pix_per_cell
            y_top = y_pos * hog_pix_per_cell
            sub_image = rescaled_image[y_top:y_top + window, x_left:x_left + window]
            # Extract HOG for this patch
            features = []
            if spatial_feat:
                features.append(get_spatial_features(sub_image, size=spatial_size))
            if hist_feat:
                features.append(get_hist_features(sub_image, nbins=hist_bin, bins_range=hist_range)[4])
            if hog_feat:
                if hog_channel == "ALL":
                    for h in hogs:
                        features.append(h[y_pos:y_pos+blocks_per_window, x_pos:x_pos+blocks_per_window].ravel())
                else:
                    features.append(hogs[hog_channel][y_pos:y_pos+blocks_per_window, x_pos:x_pos+blocks_per_window].ravel())
            # Scale features and make a prediction
            test_features = np.hstack(features).reshape(1, -1)
            test_prediction = clf.predict(test_features)
            if test_prediction == 1:
                # If prediction is true, re-calculate sliding window position
                x_box_left = np.int(x_left * rescale)
                y_box_top = np.int(y_top * rescale)
                box_size = np.int(window * rescale)
                # if box color was set, draw box on image
                if box_color is not None:
                    cv2.rectangle(draw_image, (x_box_left, y_box_top + y_start),
                                  (x_box_left + box_size, y_box_top + box_size + y_start), box_color, 6)
                # add heat to heatmap
                heatmap[y_box_top+y_start:y_box_top+box_size+y_start, x_box_left:x_box_left+box_size] += \
                        clf.decision_function(test_features)
    return draw_image, heatmap

def scan_multiple_win_sizes(image, clf, hyperparams, box_colors=None):
    """
    Scans all sliding windows of many rescaling factors, defined via hyperparams
    :param image: image to do feature sliding window classification on  
    :param clf: sklearn classification pipeline
    :param hyperparams: hyperparams specification, see hyperparams.py
    :param box_color: color to use when drawing bounding boxes on classifier hits
    :returns: returns two values: (1) the visualized classifier matches, (2) the updated heatmap 
    """
    rescales = hyperparams["RESCALES"]
    draw_image = np.zeros_like(image)
    heatmap = np.zeros(image.shape[0:2])
    for i, rescale in enumerate(rescales):
        box_color = None
        if box_colors is not None:
            box_color = box_colors[i%len(box_colors)]
        d, heatmap = scan_single_win_size(image, clf, rescale, hyperparams, box_color, heatmap)
        draw_image[(draw_image == 0).all(2)] = d[(draw_image == 0).all(2)]
    return draw_image, heatmap
    
def apply_threshold(heatmap, threshold):
    """
    Apply threshold to heatmap and return heatmap with values below threshold set to zero
    (from CarND 05 / lesson 37 Multiple Detections & False Positives)
    :param heatmap: original heatmap as x*y array
    :param threshold: threshold
    :return: the adapted heatmap with values below threshold set to zero
    """
    # Zero out pixels below the threshold
    thresh_heatmap = np.copy(heatmap)
    thresh_heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return thresh_heatmap

def draw_labeled_bboxes(image, labels, n_labels, box_color=None):
    """
    draws regions from labels to image 
    (from CarND 05 / lesson 37 Multiple Detections & False Pdositives)
    :param image: original image as x*y*3 numpy array
    :param labels: x*y numpy array label mask with each entry describing a label of that pixel
    :param n_labels: number of labels 
    :param box_color: color to draw the bounding box with 
    :return: the image with drawn bounding boxes + the list of bounding boxes in the form ((x0,y0),(x1,y1))
    """
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, n_labels + 1):
        # Find pixels with each car_number label value
        nonzero = (labels == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        bboxes.append(bbox)
    draw_image = draw_boxes(image, bboxes, box_color)
    # Return the image
    return draw_image, bboxes

def find_cars_image(image, clf, hyperparams, box_color=None, old_heatmap=None):
    """
    Finds cars in an image by applying feature extraction and sliding window classification
    (from CarND 05 / lesson 37. Multiple Detections & False Positives)
    :param image: the image to look for cars
    :param clf: the car/notcar classification pipeline applied to sliding windows
    :param hyperparams: hyperparams for feature extraction, sliding_window generation and heatmap threatment, 
       see hyperparams.py
    :param box_color: color to draw boxes with
    :param old_heatmap: an old heatmap to use as additional coefficient when finding thresholds in the image
    :return: the bounding boxes (bboxes), the images with drawn-on bounding boxes (draw_image), the
       label-mask for the image (labels_heatmap), the new heatmap (heatmap) and the aggregated heatmap (agg_heatmap)
    """
    heat_threshold = hyperparams["HEAT_THRESHOLD"]
    # Scan the image and get the new heatmap
    _, heatmap = scan_multiple_win_sizes(image, clf, hyperparams, box_colors=None)
    # Build an aggregated heatmap of this heatmap and the old heatmap
    agg_heatmap = old_heatmap+heatmap if old_heatmap is not None else heatmap
    # Apply threshold to find cars
    thresh_heatmap = apply_threshold(agg_heatmap, heat_threshold)
    # Label cars
    labels_heatmap, n_cars = label(thresh_heatmap)
    # Draw labeled boxes and get bounding boxes
    draw_image, bboxes = draw_labeled_bboxes(np.zeros_like(image), labels_heatmap, n_cars, box_color=box_color)
    # Return values
    return bboxes, draw_image, labels_heatmap, heatmap, agg_heatmap

def fancy_heatmap(heatmap, threshold):
    """
    Helper function to draw a "fancy heatmap" with red colors indicating values below and white colors above threshold
    Also, a text showing the max heatmap value will be shown
    :param heatmap: the original heatmap
    :param threshold: relevant threshold 
    :return: the "fancy heatmap"
    """
    # get some dimensions of the image, size and strength to write the text in etc.
    y = int(np.ceil(heatmap.shape[0]*.1))
    x = int(np.ceil(heatmap.shape[1]*.05))
    size = int(np.ceil(heatmap.shape[0]*.002))
    strength = int(np.ceil(heatmap.shape[0]*.004))
    # Maximum value of heatmap
    m = max(heatmap.ravel())
    # Normalized threshold (color values need to be in range 0..1)
    norm_threshold = 0 if m == 0 else threshold/m
    # Red channel will contain the heatmap, but normalized
    r = normalize(heatmap, norm='max')
    # Green and blue will also contain the heatmap, but values below (normalized) threshold set to zero
    gb = np.copy(r)
    gb[gb <= norm_threshold] = 0
    # Resulting image with all three channels merged and added text
    result = cv2.merge([r, gb, gb])
    cv2.putText(result, "Max: {:.2f}".format(m), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (1.,1.,1.), strength)
    return result

class VideoProcessor:
    """
    Class to process a video and find cars in it
    """

    def __init__(self, clf, hyperparams, box_color=None):
        """
        Init the class to process a video and find cars in it
        :param clf: classifier to use
        :param hyperparams: hyperparams for feature extraction, sliding_window generation and heatmap threatment, 
            see hyperparams.py
        :param box_color: 
        """
        self.clf = clf
        self.hyperparams = hyperparams
        self.box_color = box_color
        self.heatmap_deque = deque(maxlen=hyperparams["HEAT_FRAMES"])

    def process_image(self, image, debug=False):
        """
        Process a single frame and annotates cars / notcars
        :param image: frame to be processed for cars/notcars in x*y*3 numpy form
        :param debug: if set to True, a debug version of the annotated image will be returned with a 4-window view:
          top-left will be the output, top-right will be the current frames heatmap, bottom-left will be the 
          labels of the current frame, bottom-right will be the aggregated heatmap
        :return: the annotated image or the annotated debug image
        """
        # Get old heatmap from internal deque representation or initialize with zeros if unknown
        old_heatmap = np.zeros(image.shape[0:2]) if len(self.heatmap_deque) == 0 else sum(self.heatmap_deque)
        # Scale image from 0.0..1.0
        modified_image = np.copy(image)/255.
        # Find cars in image
        _, draw_image, labels_heatmap, new_heatmap, agg_heatmap = find_cars_image(modified_image, self.clf,
                                self.hyperparams, self.box_color, old_heatmap=old_heatmap)
        # Overlay bounding boxes on top of image
        draw_image[(draw_image == 0).all(2)] = modified_image[(draw_image == 0).all(2)]
        # Add new heatmap to deque
        self.heatmap_deque.append(new_heatmap)
        # In debug mode ...
        if debug:
            # ... create a new canvas
            result_image = np.zeros_like(image).astype(float)
            h = result_image.shape[0]
            w = result_image.shape[1]
            # get the new heatmap and aggregated heatmap as "fancy heatmap" representation
            new_heatmap_rgb = fancy_heatmap(new_heatmap, self.hyperparams["HEAT_THRESHOLD"])
            agg_heatmap_rgb = fancy_heatmap(agg_heatmap, self.hyperparams["HEAT_THRESHOLD"])
            # labels_heatmap_rgb will just be the grayscale value scaled to 0..1 and replicated across all 3 channels
            labels_heatmap_rgb = cv2.merge([normalize(labels_heatmap, norm='max')]*3)
            # place all the different images into their respective quadrant of the output canvas
            result_image[0:h//2, 0:w//2] = cv2.resize(draw_image, (w//2, h//2), interpolation=cv2.INTER_AREA)
            result_image[h//2:h, 0:w//2] = cv2.resize(labels_heatmap_rgb, (w//2, h-h//2), interpolation=cv2.INTER_AREA)
            result_image[0:h//2, w//2:w] = cv2.resize(new_heatmap_rgb, (w-w//2, h//2), interpolation=cv2.INTER_AREA)
            result_image[h//2:h, w//2:w] = cv2.resize(agg_heatmap_rgb, (w-w//2, h-h//2), interpolation=cv2.INTER_AREA)
            # rescale result to 0..255
            result_image = (result_image*255).astype(int)
        else:
            # For non-debug mode, just take scaled-back the find_cars draw_image return value
            result_image = (draw_image*255).astype(int)
        return result_image

    def process_image_debug(self, image):
        """
        Shorthand-function to call process_image(..., debug=True) so that it will work with VideoClipFile.fl_image(...)
        :param image: the image to be processed in debug mode
        :return: see process_image(...)
        """
        return self.process_image(image, debug=True)

    def process_video(self, input_path, output_path, debug=False):
        """
        Process a whole mpeg video and find cars in it.
        :param input_path: The mpeg video's file name to use as input for annotating cars
        :param output_path: The mpeg video's file name to use as output 
        :param debug: if set to True, a debug version of the annotated image will be returned with a 4-window view:
          top-left will be the output, top-right will be the current frames heatmap, bottom-left will be the 
          labels of the current frame, bottom-right will be the aggregated heatmap
        """
        clip = VideoFileClip(input_path)
        if debug:
            test_clip = clip.fl_image(self.process_image_debug)
        else:
            test_clip = clip.fl_image(self.process_image)
        test_clip.write_videofile(output_path)

def find_cars_video(input_path, output_path, clf, hyperparams, box_color=None, debug=False):
    """
    Find and annotate cars in a video.
    :param input_path: The mpeg video's file name to use as input for annotating cars
    :param output_path: The mpeg video's file name to use as output 
    :param clf: the car/notcar classification pipeline applied to sliding windows
    :param hyperparams: hyperparams for feature extraction, sliding_window generation and heatmap threatment, 
       see hyperparams.py
    :param box_color: color to draw boxes with
    :param debug: if set to True, a debug version of the annotated image will be returned with a 4-window view:
      top-left will be the output, top-right will be the current frames heatmap, bottom-left will be the 
      labels of the current frame, bottom-right will be the aggregated heatmap
    """
    v = VideoProcessor(clf, hyperparams, box_color)
    v.process_video(input_path, output_path, debug)
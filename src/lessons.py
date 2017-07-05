import numpy as np
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ntpath
from sklearn.externals import joblib
import os


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def convert_color(image, color_space):
    out_image = None
    if color_space != 'RGB':
        if color_space == 'HSV':
            out_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            out_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            out_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            out_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            out_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        out_image = np.copy(image)
    return out_image


# Helper Function to retrieve cars & non-cars
def get_templates(templates_path):
    templates = glob.glob(templates_path)
    cars = []
    notcars = []
    for template in templates:
        filename = path_leaf(template)
        if 'image' in filename or 'extra' in filename:
            notcars.append(template)
        else:
            cars.append(template)
    return cars, notcars


# 05. Manual Vehicle Detection
# Add bounding boxes in this format, these are just example coordinates.
# bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]
def draw_boxes(image, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(image)
    # draw each bounding box on your image copy using cv2.rectangle()
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes


# 09. Template Matching
# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(image, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    method = cv2.TM_CCOEFF_NORMED
    # Iterate through template list
    for template in template_list:
        # Read in templates one by one
        template_image = mpimg.imread(template).astype(np.float32)
        # Use cv2.matchTemplate() to search the image
        result = cv2.matchTemplate(image.astype(np.float32), template_image, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Determine a bounding box for the match
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


# 12. Histograms of Color
# Define a function to compute color histogram features
def get_hist_features(image, nbins=32, bins_range=(0, 256)):
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


# 16. Spatial Binning of Color
# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def get_spatial_features(image, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = convert_color(image, color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


# 19. Data Exploration
# left out intentionally


# 20. sci-kit Image HOG
# Define a function to return HOG features and visualization
def get_hog_features(image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features


# 22. Combine and Normalize Features & 29. HOG Classify
def get_features_images(images, hyperparams):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in images:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        f = get_features_image(image, hyperparams)
        features.append(f)
    # Return list of feature vectors
    return features


# 34. Search and Classify
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def get_features_image(image, hyperparams):
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
                                                     hog_cell_per_block, vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], hog_orient,
                                            hog_pix_per_cell, hog_cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)
    # Return concatenated array of features
    return np.concatenate(img_features)


# 28. Color Classify & 29. HOG Classify
def generate_classifier(templates_path, hyperparams):
    cars, notcars = get_templates(templates_path)
    # car & non-car features
    car_features = get_features_images(cars, hyperparams)
    notcar_features = get_features_images(notcars, hyperparams)
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    # Use a linear SVC
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    # Check the prediction time for a single sample
    return svc, X_test, y_test, X_scaler


def persist_classifier(clf, X_test, y_test, X_scaler, filename):
    joblib.dump((clf, X_test, y_test, X_scaler), filename)


def restore_classifier(filename):
    if os.path.isfile(filename):
        n_tuple = joblib.load(filename)
        clf = n_tuple[0]
        X_test = n_tuple[1]
        y_test = n_tuple[2]
        X_scaler = n_tuple[3]
        return clf, X_test, y_test, X_scaler
    else:
        return None


def restore_or_generate_classifier(filename, templates_path, hyperparams):
    if filename is not None:
        restored = restore_classifier(filename)
        if restored is not None:
            return restored
    return generate_classifier(templates_path, hyperparams)


# 32. Sliding Window Implementation
# Define a function that takes an image, start and stop positions in both x and y, window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(image, x_start=0, x_stop=None, y_start=0, y_stop=None,
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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


# 34. Search and Classify
# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(image, windows, clf, scaler, hyperparams):
    # Create an empty list to receive positive detection windows
    on_windows = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # Extract features for that window using single_img_features()
        features = get_features_image(test_img, hyperparams)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # Return windows for positive detections
    return on_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def scan_single_win_size(image, clf, X_scaler, rescale, hyperparams, box_color=(0,0,255), heatmap=None):
    # feature extraction hyperparams
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
    if heatmap is None:
        heatmap = np.zeros(image.shape[0:2])
    # copy image
    draw_image = np.zeros_like(image)
    rescaled_image = convert_color(image[y_start:y_stop, :, :], color_space)
    if rescale != 1:
        imshape = image.shape
        rescaled_image = cv2.resize(rescaled_image, (np.int(imshape[1] / rescale), np.int(imshape[0] / rescale)))
    # Define blocks and steps as above
    n_x_blocks = (rescaled_image.shape[1] // hog_pix_per_cell) - hog_cell_per_block + 1
    n_y_blocks = (rescaled_image.shape[0] // hog_pix_per_cell) - hog_cell_per_block + 1
    window = 64 # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    blocks_per_window = (window // hog_pix_per_cell) - hog_cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    n_x_steps = (n_x_blocks - blocks_per_window) // cells_per_step
    n_y_steps = (n_y_blocks - blocks_per_window) // cells_per_step
    hogs = []
    if hog_feat:
        for channel in range(0, rescaled_image.shape[2]):
            hogs.append(get_hog_features(rescaled_image[:, :, channel],
                        hog_orient, hog_pix_per_cell, hog_cell_per_block, feature_vec=False))
    for x_window in range(n_x_steps):
        for y_window in range(n_y_steps):
            y_pos = y_window * cells_per_step
            x_pos = x_window * cells_per_step
            x_left = x_pos * hog_pix_per_cell
            y_top = y_pos * hog_pix_per_cell
            sub_image = cv2.resize(rescaled_image[y_top:y_top + window, x_left:x_left + window], (64, 64))
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
            test_features = X_scaler.transform(np.hstack(features).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = clf.predict(test_features)
            if test_prediction == 1:
                x_box_left = np.int(x_left * rescale)
                y_box_top = np.int(y_top * rescale)
                box_size = np.int(window * rescale)
                cv2.rectangle(draw_image, (x_box_left, y_box_top + y_start),
                              (x_box_left + box_size, y_box_top + box_size + y_start), box_color, 6)
                # add heat
                heatmap[y_box_top+y_start:y_box_top+box_size+y_start, x_box_left:x_box_left+box_size] += 1.0
    return draw_image, heatmap


def scan_multiple_win_sizes(image, clf, X_scaler, hyperparams, box_colors=None, heatmap=None):
    rescales = hyperparams["RESCALES"]
    draw_image = np.zeros_like(image)
    if heatmap is None:
        heatmap = np.zeros(image.shape[0:2])
    for i, rescale in enumerate(rescales):
        box_color = None
        if box_colors is not None:
            box_color = box_colors[i%len(box_colors)]
            d, heatmap = scan_single_win_size(image, clf, X_scaler, rescale, hyperparams, box_color, heatmap)
            draw_image[(draw_image == 0).all(2)] = d[(draw_image == 0).all(2)]
    return draw_image, heatmap
    

# 37. Multiple Detections & False Positives
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


# 37. Multiple Detections & False Positives
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# 37. Multiple Detections & False Positives
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


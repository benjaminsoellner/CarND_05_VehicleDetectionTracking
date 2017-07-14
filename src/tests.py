"""
Contains tests based on matplotlib that verify the algorithms from model.py
"""

from .model import *
from .hyperparams import *
import matplotlib.pyplot as plt

def test_draw_boxes(test_image):
    """
    Draws a bunch of boxes on a test_image and shows the output with matplotlib
    :param test_image: the image to show the boxes on 
    """
    bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]
    result = draw_boxes(test_image, bboxes)
    plt.imshow(result)

def test_find_matches(test_image, template_list):
    """
    Does spacial pattern matching with a list of template files (template_list) on a test_image, 
    plots the bounding boxes of the matches with matplotlib
    :param test_image: the image to do pattern matching on
    :param template_list: the list of templates to detect in the image
    """
    bboxes = find_matches(test_image, template_list)
    result = draw_boxes(test_image, bboxes)
    plt.imshow(result)

def test_get_hist_features(test_image):
    """
    Extracts histogram features from a test_image and plots them with subplots in matplotlib
    :param test_image: the image to do histogram based feature extraction on
    """
    r, g, b, bin_centers, feature_vec = get_hist_features(test_image, nbins=32, bins_range=(0.0, 1.0))
    # Plot a figure with all three bar charts
    if r is not None and g is not None and b is not None:
        fig = plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.bar(bin_centers, r[0])
        plt.xlim(0, 1)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bin_centers, g[0])
        plt.xlim(0, 1)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bin_centers, b[0])
        plt.xlim(0, 1)
        plt.title('B Histogram')
        fig.tight_layout()
    else:
        print('Your function is returning None for at least one variable...')

def test_get_spatial_features(test_image):
    """
    Extracts spatial features from a test_image and plots them with subplots in matplotlib
    :param test_image: the image to do histogram based feature extraction on
    :return: 
    """
    feature_vec = get_spatial_features(test_image, size=(32, 32))
    # Plot features
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')

def test_get_hog_features(templates_path_pattern):
    """
    Extracts hog features from a list of templates described by templates_path_pattern and choose
    one template at random & visualize its result
    :param templates_path_pattern: the pattern to search for template images to get the features for
    :return: 
    """
    cars, notcars = get_templates(templates_path_pattern)
    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = mpimg.imread(cars[ind])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')

def test_get_features_images(templates_path_pattern):
    """
    Extracts all features from a list of templates described by templates_path_pattern and choose
    one template at random & visualize its result
    :param templates_path_pattern: the pattern to search for template images to get the hog features for
    :return: 
    """
    global HYPERPARAMS
    cars, notcars = get_templates(templates_path_pattern)
    car_features = get_features_images(cars, hyperparams=HYPERPARAMS)
    notcar_features = get_features_images(notcars, hyperparams=HYPERPARAMS)
    if len(car_features) > 0 or len(notcar_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(cars))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
    else:
        print('Your function only returns empty feature vectors...')

def test_generate_classifier(templates_path_pattern, persist_clf_filename):
    """
    Generate and store the car/notcar classifier based on the car/notcar templates
    :param templates_path_pattern: filename pattern where to find car/notcar images
    :param persist_clf_filename: pickle file to persist the classifier with train and test set
       persisting will be skipped if the file already exists
    """
    global HYPERPARAMS
    clf, X_test, y_test = generate_classifier(templates_path_pattern, hyperparams=HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    n_predict = 10
    print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    if persist_clf_filename is not None:
        persist_classifier(clf, X_test, y_test, pickle_file=persist_clf_filename)

def test_slide_window(test_image):
    """
    Generate sliding windows over the image with 128x128 window size and 50% window overlap in x&y,
    visualize the sliding windows with matplotlib.
    :param test_image: the image to generate the sliding windows for 
    """
    windows = slide_window(test_image, xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    window_img = draw_boxes(test_image, windows, color=(0., 0., 1.), thick=6)
    plt.imshow(window_img)

def test_search_windows(templates_path_pattern, test_image, restore_clf_filename):
    """
    Search sliding windows of size 96x96 with 50% overlap for images found from the car/notcar classifier
    :param templates_path_pattern: filename pattern where to find car/notcar images
    :param test_image: the test image to do the template matching on
    :param restore_clf_filename: the filename of the persisted classifier, deserialized if file is present, otherwise
      used for serialization if the classifier has to be generated/trained first
    """
    global HYPERPARAMS
    # Generate and test classifier
    clf, X_test, y_test = restore_or_generate_classifier(restore_clf_filename, templates_path_pattern, HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    # Generate slidewindows
    windows = slide_window(test_image, xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    # find "hot matches"
    draw_image = np.copy(test_image)
    hot_windows, confidences = search_windows(test_image, windows, clf, hyperparams=HYPERPARAMS)
    window_img = draw_boxes(draw_image, hot_windows, color=(0., 0., 1.), thick=6)
    print("Confidences:")
    print(confidences)
    plt.imshow(window_img)

def test_scan_single_win_size(templates_path, test_image, restore_clf_filename):
    """
    Search sliding windows of size 64x64 with 50% overlap for images found from the car/notcar classifier.
    Use performant version of feature extraction with hog features only generated once.
    :param templates_path: filename pattern where to find car/notcar images
    :param test_image: the test image to do the template matching on
    :param restore_clf_filename: the filename of the persisted classifier, deserialized if file is present, otherwise
      used for serialization if the classifier has to be generated/trained first
    """
    global HYPERPARAMS
    # Generate and test classifier
    clf, X_test, y_test = restore_or_generate_classifier(restore_clf_filename, templates_path, HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    draw_image, heatmap = scan_single_win_size(test_image, clf, HYPERPARAMS["RESCALES"][0], hyperparams=HYPERPARAMS)
    draw_image[(draw_image  == 0).all(2)] = test_image[(draw_image == 0).all(2)]
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_image)
    plt.title('Boxes Found')
    plt.subplot(122)
    plt.imshow(heatmap/max(heatmap.ravel()), cmap='hot')
    plt.title('Heatmap')
    fig.tight_layout()

def test_scan_multiple_win_sizes(templates_path_pattern, test_image, restore_clf_filename):
    """
    Search multiple sliding windows sizes (according to hyperparameter specification) with for images 
    found from the car/notcar classifier. Use performant version of feature extraction with hog features only generated 
    once.
    :param templates_path_pattern: filename pattern where to find car/notcar images
    :param test_image: the test image to do the template matching on
    :param restore_clf_filename: the filename of the persisted classifier, deserialized if file is present, otherwise
      used for serialization if the classifier has to be generated/trained first
    """
    global HYPERPARAMS
    # Generate and test classifier
    clf, X_test, y_test = restore_or_generate_classifier(restore_clf_filename, templates_path_pattern, HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    draw_image, heatmap = scan_multiple_win_sizes(test_image, clf, hyperparams=HYPERPARAMS,
                                         box_colors=[(0.,0.,1.),(0.,1.,0.),(1.,0.,0.)])
    draw_image[(draw_image  == 0).all(2)] = test_image[(draw_image == 0).all(2)]
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_image)
    plt.title('Boxes Found')
    plt.subplot(122)
    plt.imshow(heatmap/max(heatmap.ravel()), cmap='hot')
    plt.title('Heatmap')
    fig.tight_layout()

def test_find_cars_image(templates_path_pattern, test_image, restore_clf_filename):
    """
    Search multiple sliding windows sizes (according to hyperparameter specification) for images 
    found from the car/notcar classifier, cluster the hits into bounding boxes and label them appropriately.
    :param templates_path_pattern: filename pattern where to find car/notcar images
    :param test_image: the test image to do the template matching on
    :param restore_clf_filename: the filename of the persisted classifier, deserialized if file is present, otherwise
      used for serialization if the classifier has to be generated/trained first
    """
    global HYPERPARAMS
    clf, X_test, y_test, X_scaler = restore_or_generate_classifier(restore_clf_filename, templates_path_pattern, HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    bboxes, draw_image, labels_heatmap, heatmap = find_cars_image(test_image, clf, X_scaler, HYPERPARAMS, box_color=(0.,0.,1.))
    print('Bounding boxes found: ', bboxes)
    draw_image[(draw_image == 0).all(2)] = test_image[(draw_image == 0).all(2)]
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(labels_heatmap, cmap='gray')
    plt.title('Labels')
    plt.subplot(122)
    plt.imshow(draw_image)
    plt.title('Bounding Boxes')
    fig.tight_layout()

def test_find_cars_video(templates_path, test_video_input, test_video_output, restore_clf_filename, debug=False):
    """
    Search video with sliding windows sizes (according to hyperparameter specification) for images 
    found from the car/notcar classifier, cluster the hits into bounding boxes and label them appropriately.
    :param templates_path_pattern: filename pattern where to find car/notcar images
    :param test_video_input: the video to process for cars / notcars
    :param test_video_output: the video to output the labeled cars/notcars
    :param restore_clf_filename: the filename of the persisted classifier, deserialized if file is present, otherwise
      used for serialization if the classifier has to be generated/trained first
    :param debug: if set to True, uses debug mode with verbose video output drawing heatmaps and labels into the video
    """
    global HYPERPARAMS
    clf, X_test, y_test = restore_or_generate_classifier(restore_clf_filename, templates_path, HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    find_cars_video(test_video_input, test_video_output, clf, HYPERPARAMS, box_color=(0.,0.,1.), debug=debug)
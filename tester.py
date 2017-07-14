"""
The tester for the "Car/Notcar Vehicle Tracker": Test Script to make sure the 
algorithm works properly.
from the Udacity Self Driving Car Engineer Nanodegree.
by Benjamin Söllner, July 2017

Usage:
  python tester.py (draw_boxes|find_matches|get_hist_features|
                    get_spatial_features|get_hog_features|get_features_images|
                    generate_classifier|slide_window|search_windows|
                    scan_single_win_size|scan_multiple_win_sizes|
                    find_cars_image|find_cars_video|find_cars_video_debug)
"""

from src.tests import *
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parameter handling: the first command line param should be used to supply the test case
    if len(sys.argv) == 1:
        print(__doc__)
        exit(0)
    test_case = sys.argv[1]
    print("Testing " + test_case)
    # matplotlib interactive mode off
    plt.ioff()

    # Set test images
    templates_path = 'data/**/**/*.png'
    test_image = mpimg.imread('submodules/CarND-Vehicle-Detection/test_images/test1.jpg')/255.
    template_list = [
        'data/vehicles/GTI_MiddleClose/image0000.png',
        'data/vehicles/GTI_MiddleClose/image0039.png',
        'data/vehicles/GTI_MiddleClose/image0054.png',
        'data/vehicles/GTI_MiddleClose/image0073.png',
        'data/vehicles/GTI_MiddleClose/image0084.png'
    ]
    template_image = mpimg.imread(template_list[4])
    clf_pickle = "pickles/classifier.pkl"

    # Set test videos
    # test_video_input = "submodules/CarND-Vehicle-Detection/test_video.mp4"
    # test_video_output = "test_video_output.mp4"
    # test_video_output_debug = "test_video_output_debug.mp4"
    test_video_input = "submodules/CarND-Vehicle-Detection/project_video.mp4"
    test_video_output = "project_video_output.mp4"
    test_video_output_debug = "project_video_output_debug.mp4"

    if test_case == "draw_boxes":
        # Draws a bunch of boxes on a test_image and shows the output with matplotlib
        test_draw_boxes(test_image)

    elif test_case == "find_matches":
        # Does spacial pattern matching with a list of template files (template_list) on a test_image,
        # plots the bounding boxes of the matches with matplotlib
        test_find_matches(test_image, template_list)

    elif test_case == "get_hist_features":
        # Extracts histogram features from a test_image and plots them with subplots in matplotlib
        test_get_hist_features(template_image)

    elif test_case == "get_spatial_features":
        # Extracts spatial features from a test_image and plots them with subplots in matplotlib
        test_get_spatial_features(template_image)

    elif test_case == "get_hog_features":
        # Extracts hog features from a list of templates described by templates_path_pattern and choose
        # one template at random & visualize its result
        test_get_hog_features(templates_path)

    elif test_case == "get_features_images":
        # Extracts all features from a list of templates described by templates_path_pattern and choose
        # one template at random & visualize its result
        test_get_features_images(templates_path)

    elif test_case == "generate_classifier":
        # Generate and store the car/notcar classifier based on the car/notcar templates
        test_generate_classifier(templates_path, persist_clf_filename=clf_pickle)

    elif test_case == "slide_window":
        # Generate sliding windows over the image with 128x128 window size and 50% window overlap in x&y,
        # visualize the sliding windows with matplotlib.
        test_slide_window(test_image)

    elif test_case == "search_windows":
        # Search sliding windows of size 96x96 with 50% overlap for images found from the car/notcar classifier
        test_search_windows(templates_path, test_image, restore_clf_filename=clf_pickle)

    elif test_case == "scan_single_win_size":
        # Search sliding windows of size 64x64 with 50% overlap for images found from the car/notcar classifier.
        # Use performant version of feature extraction with hog features only generated once.
        test_scan_single_win_size(templates_path, test_image, restore_clf_filename=clf_pickle)

    elif test_case == "scan_multiple_win_sizes":
        # Search multiple sliding windows sizes (according to hyperparameter specification) with for images
        # found from the car/notcar classifier. Use performant version of feature extraction with hog features only
        # generated once.
        test_scan_multiple_win_sizes(templates_path, test_image, restore_clf_filename=clf_pickle)

    elif test_case == "find_cars_image":
        # Search multiple sliding windows sizes (according to hyperparameter specification) for images
        # found from the car/notcar classifier, cluster the hits into bounding boxes and label them appropriately.
        test_find_cars_image(templates_path, test_image, restore_clf_filename=clf_pickle)

    elif test_case == "find_cars_video":
        # Search video with sliding windows sizes (according to hyperparameter specification) for images
        # found from the car/notcar classifier, cluster the hits into bounding boxes and label them appropriately.
        test_find_cars_video(templates_path, test_video_input, test_video_output, restore_clf_filename=clf_pickle)

    elif test_case == "find_cars_video_debug":
        # Search video with sliding windows sizes (according to hyperparameter specification) for images
        # found from the car/notcar classifier, cluster the hits into bounding boxes and label them appropriately.
        # Uses debug mode with verbose video output drawing heatmaps and labels into the video
        test_find_cars_video(templates_path, test_video_input, test_video_output_debug, restore_clf_filename=clf_pickle, debug=True)

    plt.show()

from src.tests import *
import sys
import cv2
import matplotlib.pyplot as plt
import msvcrt as m

if __name__ == "__main__":
    plt.ioff()
    # Set up easy test files
    templates_path = 'data/**/**/*.png'
    test_image = mpimg.imread('submodules/CarND-Vehicle-Detection/test_images/test1.jpg')/255.
    template_list = [
        'data/vehicles/GTI_MiddleClose/image0000.png',
        'data/vehicles/GTI_MiddleClose/image0039.png',
        'data/vehicles/GTI_MiddleClose/image0054.png',
        'data/vehicles/GTI_MiddleClose/image0073.png',
        'data/vehicles/GTI_MiddleClose/image0084.png'
    ]
    clf_pickle = "pickles/classifier.pkl"
    #test_video_input = "submodules/CarND-Vehicle-Detection/test_video.mp4"
    #test_video_output = "test_video_output.mp4"
    #test_video_output_debug = "test_video_output_debug.mp4"
    test_video_input = "submodules/CarND-Vehicle-Detection/project_video.mp4"
    test_video_output = "project_video_output.mp4"
    test_video_output_debug = "project_video_output_debug.mp4"
    template_image = mpimg.imread(template_list[4])
    test_case = sys.argv[1] if len(sys.argv) > 0 else "draw_boxes"
    print("Testing " + test_case)
    if test_case == "draw_boxes":
        test_draw_boxes(test_image)
    elif test_case == "find_matches":
        test_find_matches(test_image, template_list)
    elif test_case == "get_hist_features":
        test_get_hist_features(template_image)
    elif test_case == "get_spatial_features":
        test_get_spatial_features(template_image)
    elif test_case == "get_hog_features":
        test_get_hog_features(templates_path)
    elif test_case == "get_features":
        test_get_features_images(templates_path)
    elif test_case == "generate_classifier":
        test_generate_classifier(templates_path, persist_clf_filename=clf_pickle)
    elif test_case == "slide_window":
        test_slide_window(test_image)
    elif test_case == "search_windows":
        test_search_windows(templates_path, test_image, restore_clf_filename=clf_pickle)
    elif test_case == "scan_single_win_size":
        test_scan_single_win_size(templates_path, test_image, restore_clf_filename=clf_pickle)
    elif test_case == "scan_multiple_win_sizes":
        test_scan_multiple_win_sizes(templates_path, test_image, restore_clf_filename=clf_pickle)
    elif test_case == "find_cars_image":
        test_find_cars_image(templates_path, test_image, restore_clf_filename=clf_pickle)
    elif test_case == "find_cars_video":
        test_find_cars_video(templates_path, test_video_input, test_video_output, restore_clf_filename=clf_pickle)
    elif test_case == "find_cars_video_debug":
        test_find_cars_video(templates_path, test_video_input, test_video_output_debug, restore_clf_filename=clf_pickle, debug=True)
    plt.show()

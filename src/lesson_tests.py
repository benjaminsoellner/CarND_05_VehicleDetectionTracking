import matplotlib.pyplot as plt
from .lessons import *
from .hyperparams import *


def test_draw_boxes(test_image):
    bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]
    result = draw_boxes(test_image, bboxes)
    plt.imshow(result)


def test_find_matches(test_image, template_list):
    bboxes = find_matches(test_image, template_list)
    result = draw_boxes(test_image, bboxes)
    plt.imshow(result)


def test_get_hist_features(test_image):
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
    feature_vec = get_spatial_features(test_image, color_space='RGB', size=(32, 32))
    # Plot features
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')


def test_get_hog_features(templates_path):
    cars, notcars = get_templates(templates_path)
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


def test_get_features_images(templates_path):
    global HYPERPARAMS
    cars, notcars = get_templates(templates_path)
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


def test_generate_classifier(templates_path, persist_clf_filename):
    global HYPERPARAMS
    svc, X_test, y_test, X_scaler = generate_classifier(templates_path, hyperparams=HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    if persist_clf_filename is not None:
        persist_classifier(svc, X_test, y_test, X_scaler, filename=persist_clf_filename)


def test_slide_window(test_image):
    windows = slide_window(test_image, xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    window_img = draw_boxes(test_image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)


def test_search_windows(templates_path, test_image, restore_clf_filename):
    global HYPERPARAMS
    # Generate and test classifier
    svc, X_test, y_test, X_scaler = restore_or_generate_classifier(restore_clf_filename, templates_path, HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Generate slidewindows
    windows = slide_window(test_image, xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    # find "hot matches"
    draw_image = np.copy(test_image)
    hot_windows = search_windows(test_image, windows, svc, X_scaler, hyperparams=HYPERPARAMS)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)


def test_scan_single_win_size(templates_path, test_image, restore_clf_filename):
    global HYPERPARAMS
    # Generate and test classifier
    svc, X_test, y_test, X_scaler = restore_or_generate_classifier(restore_clf_filename, templates_path, HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    draw_image, heatmap = scan_single_win_size(test_image, svc, X_scaler, HYPERPARAMS["RESCALES"][0], hyperparams=HYPERPARAMS)
    draw_image[(draw_image  == 0).all(2)] = test_image[(draw_image == 0).all(2)]
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_image)
    plt.title('Boxes Found')
    plt.subplot(122)
    plt.imshow(heatmap/max(heatmap.ravel()), cmap='hot')
    plt.title('Heatmap')
    fig.tight_layout()


def test_scan_multiple_win_sizes(templates_path, test_image, restore_clf_filename):
    global HYPERPARAMS
    # Generate and test classifier
    svc, X_test, y_test, X_scaler = restore_or_generate_classifier(restore_clf_filename, templates_path, HYPERPARAMS)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    draw_image, heatmap = scan_multiple_win_sizes(test_image, svc, X_scaler, hyperparams=HYPERPARAMS,
                                         box_colors=[(0,0,255),(0,255,0),(255,0,0)])
    draw_image[(draw_image  == 0).all(2)] = test_image[(draw_image == 0).all(2)]
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_image)
    plt.title('Boxes Found')
    plt.subplot(122)
    plt.imshow(heatmap/max(heatmap.ravel()), cmap='hot')
    plt.title('Heatmap')
    fig.tight_layout()
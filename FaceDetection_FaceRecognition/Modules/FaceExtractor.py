from Modules.Helper import ImageProcessor
from Modules.Helper import FaceLocalisator
from Modules.Helper import FaceOperator

from Modules.Enums import FaceLocalisationModes
from Modules.Enums import ImageProcessingModes


# specifies if actions and times should be logged or not
_is_logging = False
# counts how often a face is found at second attempt
_second_chance_counter  = 0


def _image_processing(img, processing_methods):
    """
    Given an array of dicts with method-name and method-parameters this function
    applies these on the given image and returns the manipulated image.
    Used for pre-processing as well as post-processing.

    Array format example: [ {ImageProcessingModes.GRAYSCALE_CONVERTION},
                            {ImageProcessingModes.CLAHE: {'cliplimit': 4, 'tile_grid_size': (8, 8)]

    :param img:                 image on which to apply an given processes
    :param processing_methods:  Array-like object with methods and parameters.
                                Array format example: [ {ImageProcessingModes.GRAYSCALE_CONVERTION},
                                                        {ImageProcessingModes.CLAHE: {'cliplimit': 4, 'tile_grid_size': (8, 8)]
    :return:                    manipulated image
    """
    for process in processing_methods:

        if ImageProcessingModes.CLAHE in process:
            if _is_logging:
                print('ImageProcessor: CLAHE')
            cliplimit       = None
            tile_grid_size  = None
            try:
                cliplimit       = process[ImageProcessingModes.CLAHE]['cliplimit']
            except KeyError:
                print('INFO: FaceExtractor._image_processing(): parameter cliplimit not defined')
            try:
                tile_grid_size  = process[ImageProcessingModes.CLAHE]['tile_grid_size']
            except KeyError:
                print('INFO: FaceExtractor._image_processing(): parameter tile_grid_Size not defined')
            img = ImageProcessor.clahe(img,
                                       cliplimit=cliplimit,
                                       tile_grid_size=tile_grid_size)

        elif ImageProcessingModes.GAMMA_CORRECTION in process:
            if _is_logging:
                print('ImageProcessor: gamma correction')
            gamma = None
            try:
                gamma = process[ImageProcessingModes.GAMMA_CORRECTION]['gamma']
            except KeyError:
                print('INFO: FaceExtractor._image_processing(): parameter gamma not defined')
            img = ImageProcessor.gamma_correction(img,
                                                  gamma=gamma)

        elif ImageProcessingModes.GRAYSCALE_CONVERTION in process:
            if _is_logging:
                print('ImageProcessor: grayscale convertion')
            img = ImageProcessor.img2gray(img)

        elif ImageProcessingModes.HISTOGRAM_EQUALIZATION in process:
            if _is_logging:
                print('ImageProcessor: histogram equalization')
            img = ImageProcessor.histogram_equalization(img)

        else:
            print('WARNING: FaceExtractor._image_processing(): processing method unknown:\n', process)

    return img


def _face_localization(img, localization_method):
    """
    Uses given method to localise faces on given image.

    :param img:                     grayscale image on which to search for faces
    :param localization_method:     method and parameters for localization
                                    Example without param min_neighbors:
                                    [{FaceLocalisationModes.HAARCASCADES_FACE_PRE_TRAINED:
                                    {'scale_factor': 2, 'max_faces': 5}}]
    :return:                        list of faces
    """
    if FaceLocalisationModes.HAARCASCADES_FACE_PRE_TRAINED in localization_method[0]:
        if _is_logging:
            print('FaceLocalization: haarcascades')
        scale_factor    = None
        min_neighbors   = None
        max_faces       = None
        try:
            scale_factor    = localization_method[0][FaceLocalisationModes.HAARCASCADES_FACE_PRE_TRAINED]['scale_factor']
        except KeyError:
            print('INFO: FaceExtractor._face_localization(): parameter scale_factor not defined')
        try:
            min_neighbors   = localization_method[0][FaceLocalisationModes.HAARCASCADES_FACE_PRE_TRAINED]['min_neighbors']
        except KeyError:
            print('INFO: FaceExtractor._face_localization(): parameter min_neighbors not defined')
        try:
            max_faces       = localization_method[0][FaceLocalisationModes.HAARCASCADES_FACE_PRE_TRAINED]['max_faces']
        except KeyError:
            print('INFO: FaceExtractor._face_localization(): parameter max_faces not defined')

        return FaceLocalisator.haarcascades_detection(img,
                                                      scale_factor=scale_factor,
                                                      min_neighbors=min_neighbors,
                                                      max_faces=max_faces)

    elif FaceLocalisationModes.HOG_PRE_TRAINED in localization_method[0]:
        if _is_logging:
            print('FaceLocalization: HoG')
        max_faces   = None
        up_sampling = None
        try:
            max_faces   = localization_method[0][FaceLocalisationModes.HOG_PRE_TRAINED]['max_faces']
        except KeyError:
            print('INFO: FaceExtractor._face_localization(): parameter max_faces not defined')
        try:
            up_sampling = localization_method[0][FaceLocalisationModes.HOG_PRE_TRAINED]['up_sampling']
        except KeyError:
            print('INFO: FaceExtractor._face_localization(): parameter up_sampling not defined')
        return FaceLocalisator.hog_detection(img,
                                             max_faces=max_faces,
                                             up_sampling=up_sampling)
    else:
        print('WARNING: FaceExtractor.extract_faces(): no face localisation method selected!')


def extract_faces(img, pre_processing_methods, localization_method,
                  post_processing_methods=False, face_out_size=(224, 224)):
    """
    Takes given image and applies all given pre-processing steps before
    it tries to find all faces on this image using given method.
    If no face is found, post-processing steps are applied if given and
    face localization is tried again.
    If the number of faces exceeds the given number of max_faces,
    the list of faces is cut down to fit this maximum number of faces.
    This is done by removing the smallest faces from the list until the given number is reached,
    as smaller faces are considered less important.
    Finally the remaining list of faces is resized to given output size and returned.


    :param img:                         rgb/grayscale image
    :param pre_processing_methods:      array of pre-processing methods with relevant parameters
    :param localization_method:         array with one element of used localization method and appropriate parameters
    :param post_processing_methods:     array of post-processing methods with relevant parameters
    :param face_out_size:               tuple of (width, height), defines output size of each face
    :return:                            list of grayscale faces found on the image, resized to given size and
                                        manipulated with all given processing steps
    """
    global _is_logging, _second_chance_counter
    start_time = None
    if _is_logging:
        import time
        print('STARTING FACE EXTRACTION')
        start_time = time.time()
        total_time = time.time()

    # PRE PROCESSING
    img = _image_processing(img, pre_processing_methods)
    if _is_logging:
        print('pre-processing:\t\t', time.time() - start_time, 's')
        start_time = time.time()

    # FACE LOCALISATION
    faces = _face_localization(img, localization_method)
    if _is_logging:
        print('first localisation:\t', time.time()-start_time, 's')
        if len(faces) > 0:
                print('second localization canceled')
        start_time = time.time()

    # POST PROCESS IF NO FACE FOUND
    if post_processing_methods and len(faces) < 1:
        img     = _image_processing(img, post_processing_methods)
        if _is_logging:
            print('post-processing:\t\t:', time.time()-start_time, 's')
            start_time = time.time()
        faces   = _face_localization(img, localization_method)
        if len(faces) > 0:
            print('INFO: Face found in second attempt')
            _second_chance_counter += 1
        if _is_logging:
            print('second localization:\t:', time.time() - start_time, 's')
            start_time = time.time()

    # NO FACE FOUND, RETURN NONE
    if len(faces) < 1:
        print('INFO: FaceExtractor: No face found!')
        if _is_logging:
            print('---------------\n'
                  'no face found\n'
                  'total time:', time.time()-total_time, 's',
                  '\n---------------')
        return None

    # RESIZE FACES TO GIVEN SIZE AND RETURN
    else:
        faces_resized = list()
        for face in faces:
            faces_resized.append(FaceOperator.scale(face, face_out_size[0], face_out_size[1]))
        if _is_logging:
            print('resize faces:\t\t', time.time()-start_time, 's')
            print('---------------\n'
                  'faces found:', len(faces_resized),
                  '\ntotal time:', time.time() - total_time, 's',
                  '\n---------------')
        return faces_resized


def set_logging(b):
    """
    True:   log actions and time
    False:  do not log anything

    :param b: boolean
    """
    global _is_logging
    if b:
        _is_logging = True
    if not b:
        _is_logging = False


def reset_second_chance_counter():
    global _second_chance_counter
    _second_chance_counter = 0


def get_second_chance_counter():
    global _second_chance_counter
    return _second_chance_counter

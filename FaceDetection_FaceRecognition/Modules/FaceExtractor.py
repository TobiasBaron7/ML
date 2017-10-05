from Modules.Helper import ImageProcessor
from Modules.Helper import FaceLocalisator
from Modules.Helper import FaceOperator

from Modules.Enums import FaceLocalisationModes
from Modules.Enums import ImageProcessingModes


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

        # TODO
        elif ImageProcessingModes.FRONTALIZATION in process:
            print('frontalize')

        elif ImageProcessingModes.GAMMA_CORRECTION in process:
            gamma = None
            try:
                gamma = process[ImageProcessingModes.GAMMA_CORRECTION]['gamma']
            except KeyError:
                print('INFO: FaceExtractor._image_processing(): parameter gamma not defined')
            img = ImageProcessor.gamma_correction(img,
                                                  gamma=gamma)

        elif ImageProcessingModes.GRAYSCALE_CONVERTION in process:
            img = ImageProcessor.img2gray(img)

        elif ImageProcessingModes.HISTOGRAM_EQUALIZATION in process:
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
                  post_processing_methods, face_out_size=(224, 224)):
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
    # PRE PROCESSING
    img = _image_processing(img, pre_processing_methods)

    # FACE LOCALISATION
    faces = _face_localization(img, localization_method)

    # POST PROCESS IF NO FACE FOUND
    if len(faces) < 1:
        img     = _image_processing(img, post_processing_methods)
        faces   = _face_localization(img, localization_method)

    # NO FACE FOUND, RETURN NONE
    if len(faces) < 1:
        return None

    # RESIZE FACES TO GIVEN SIZE AND RETURN
    else:
        faces_resized = list()
        for face in faces:
            faces_resized.append(FaceOperator.scale(face, face_out_size[0], face_out_size[1]))
        return faces_resized


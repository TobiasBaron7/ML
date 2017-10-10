import Modules.FaceExtractor as FExtractor
import Modules.FaceFeature as FFeature
import cv2
from Modules.Enums import ImageProcessingModes as Ip
from Modules.Enums import FaceLocalisationModes as Fl
from Modules.Enums import FeatureExtractionModes as Fe


# General settings
image_path                  = 'data/test/family.jpg'

# Test-Settings
# define which module to test
# when testing later modules, previous modules are needed an thus used with provided settings as well

# FaceExtracor
test_FaceExtractor          = False
log_FaceExtractor           = True

# Image pre-processing
pre_processing              = [Ip.GRAYSCALE_CONVERTION,
                               Ip.CLAHE]
pre_gamma                   = 2
pre_cliplimit               = 5
pre_tile_grid_size          = (8, 8)

# Image post-processing
post_processing             = [Ip.GAMMA_CORRECTION]
post_gamma                  = 3
post_cliplimit              = 8
post_tile_grid_size         = (5, 5)

# FaceLocalisator
method_face_localization    = Fl.HOG_PRE_TRAINED
max_faces                   = 100
up_sampling                 = 1     # hog
scale_factor                = 1.3   # haarcascades
min_neighbors               = 5     # haarcascades

# FaceOperator
face_out_width              = 224   # 224x224 needed when using CNN_VGG_16_PRE_TRAINED
face_out_height             = face_out_width
face_out_size               = (face_out_width, face_out_height)


# FaceFeature
test_FaceFeature            = True
log_FaceFeature             = True
method_face_feature         = Fe.CNN_VGG_16_PRE_TRAINED

if __name__ == '__main__':
    face_list       = None
    feature_list    = list()

    if test_FaceExtractor or test_FaceFeature:
        img = cv2.imread(image_path)
        pre_processing_methods  = [None] * len(pre_processing)
        post_processing_methods = [None] * len(post_processing)
        localization_method     = []

        for i in range(len(pre_processing_methods)):
            if pre_processing[i] is Ip.GRAYSCALE_CONVERTION:
                pre_processing_methods[i] = {pre_processing[i]}
            elif pre_processing[i] is Ip.HISTOGRAM_EQUALIZATION:
                pre_processing_methods[i] = {pre_processing[i]}
            elif pre_processing[i] is Ip.GAMMA_CORRECTION:
                pre_processing_methods[i] = {pre_processing[i]: {'gamma': pre_gamma}}
            elif pre_processing[i] is Ip.CLAHE:
                pre_processing_methods[i] = {pre_processing[i]: {'cliplimit': pre_cliplimit, 'tile_grid_size': pre_tile_grid_size}}

        for i in range(len(post_processing_methods)):
            if post_processing[i] is Ip.GRAYSCALE_CONVERTION:
                post_processing_methods[i] = {post_processing[i]}
            elif post_processing[i] is Ip.HISTOGRAM_EQUALIZATION:
                post_processing_methods[i] = {post_processing[i]}
            elif post_processing[i] is Ip.GAMMA_CORRECTION:
                post_processing_methods[i] = {post_processing[i]: {'gamma': post_gamma}}
            elif post_processing[i] is Ip.CLAHE:
                post_processing_methods[i] = {post_processing[i]: {'cliplimit': post_cliplimit, 'tile_grid_size': post_tile_grid_size}}

        if method_face_localization is Fl.HOG_PRE_TRAINED:
            localization_method = [{method_face_localization: {'up_sampling': up_sampling, 'max_faces': max_faces}}]
        elif method_face_localization is Fl.HAARCASCADES_FACE_PRE_TRAINED:
            localization_method = [{method_face_localization: {'scale_factor': scale_factor, 'max_faces': max_faces}}]

        FExtractor.set_logging(log_FaceExtractor)
        face_list = FExtractor.extract_faces(img, pre_processing_methods, localization_method, post_processing_methods, face_out_size)

        if test_FaceExtractor:
            face_counter = 0
            for face in face_list:
                cv2.imshow(str(face_counter), face)
                face_counter += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if test_FaceFeature:
        FFeature.set_logging(log_FaceFeature)
        for face in face_list:
            feature = FFeature.extract_features(face, mode=method_face_feature)
            if feature.any():
                feature_list.append(feature)
        print('feature_list length:', len(feature_list))


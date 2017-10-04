class DistanceModes:
    EUCLIDEAN                       = 0
    CITY_MAP                        = 1
    WEIGHTED_X_SQUARE               = 2


class ErrorCode:
    OK                              = 0
    INPUT_NULL                      = 1
    OUTPUT_NULL                     = 2
    UNKNOWN_ERROR                   = 3
    NO_FACE                         = 4


class ImageProcessingModes:
    GRAYSCALE_CONVERTION            = 0
    HISTOGRAM_EQUALIZATION          = 1
    CLAHE                           = 2
    GAMMA_CORRECTION                = 3
    FRONTALIZATION                  = 4


class FaceLocalisationModes:
    HAARCASCADES_FACE_PRE_TRAINED   = 0
    HOG_PRE_TRAINED                 = 1


class FeatureExtractionModes:
    CNN_VGG_16_PRE_TRAINED          = 0


class IdentificationModes:
    NEAREST_NEIGHBOUR               = 0
    K_NEAREST_NEIGHBOUR             = 1


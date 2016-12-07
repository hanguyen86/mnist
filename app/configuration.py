"""
Handwritten Digit Recognition
Licence: BSD
Author : Hoang Anh Nguyen
"""

class Config(object):
    """
    Configuration base, for all environments.
    """
    DEBUG               = False
    TESTING             = False
    SECRET_KEY          = "MINHACHAVESECRETA"
    CSRF_ENABLED        = True
    CLASSIFIER_TYPE     = "CNNClassifier"  # 'SoftmaxClassifier' or 'CNNClassifier'
    UPLOAD_FOLDER       = './uploads'
    ALLOWED_EXTENSIONS  = set(['png', 'jpg', 'jpeg'])

class ProductionConfig(Config):
    CLASSIFIER_TYPE = "CNNClassifier"

class DevelopmentConfig(Config):
    DEBUG = True
    CLASSIFIER_TYPE = "SoftmaxClassifier"

class TestingConfig(Config):
    TESTING = True
    CLASSIFIER_TYPE = "SoftmaxClassifier"
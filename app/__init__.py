"""
Handwritten Digit Recognition
Licence: BSD
Author : Hoang Anh Nguyen
"""

from flask import Flask
from flask.ext.bootstrap import Bootstrap

app = Flask(__name__)

#Configuration of application, see configuration.py, choose one and uncomment.
#app.config.from_object('configuration.ProductionConfig')
app.config.from_object('app.configuration.DevelopmentConfig')
#app.config.from_object('configuration.TestingConfig')

bs = Bootstrap(app) #flask-bootstrap

from app import mnist
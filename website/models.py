from sqlalchemy_imageattach.entity import Image, image_attachment
from sqlalchemy.ext.declarative import declarative_base
from . import db 

Base = declarative_base()

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    img = image_attachment('imageurl')

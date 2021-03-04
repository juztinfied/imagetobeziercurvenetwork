from . import db 

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.BLOB)
    
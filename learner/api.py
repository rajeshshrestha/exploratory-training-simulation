import logging

from flask import (Flask)
from flask_cors import CORS
from flask_restful import (Api)
from rich.console import Console
from utils.import_info import Import
from utils.feedback import Feedback
from utils.sample import Sample

console = Console()

'''Get logger'''
logger = logging.getLogger(__file__)

# Flask configs
app = Flask(__name__)
CORS(app,
     resources={
         r"/*": {
             "origins": "*"
         }
     })
api = Api(app)


api.add_resource(Import,
                 '/duo/api/import',
                 resource_class_kwargs={
                 })

api.add_resource(Feedback,
                 '/duo/api/feedback',
                 resource_class_kwargs={
                 })
api.add_resource(Sample,
                 '/duo/api/sample',
                 resource_class_kwargs={
                 })

if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=5000)

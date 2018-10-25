#!/usr/bin/python3
# coding: utf-8
"""
Clase que hace deploy a ML Engine.
"""
import os
import logging
import subprocess
import requests
import re
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.saved_model import tag_constants
import keras.backend as K
from keras.models import load_model
from google.cloud import storage

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    #filename='log.txt'
)

class DeployMLEngine(object):
    """
    Clase que hace deploy a ML Engine.
    Esta clase ayuda a hacer el despliegue de nuevos modelos pre-entrenados a ML Engine
    """
    def __init__(self, fname, model_name, project_id, version_name, model_dest):
        """"
        Args:
            fname(str): ruta del archivo donde está guardado el modelo de Keras
            version_name(string): versión del modelo
            model_dest(string): ubicación donde está guardada la versión del modelo
            project_id(string): id del proyecto que hospedará el modelo
            model_name(string): nombre del modelo
        """
        self.fname = fname
        self.model_name = model_name
        self.project_id = project_id
        self.version_name = version_name
        self.model_dest = model_dest

    def upload_blob(self):
        """Sube el archivo h5 (original) a un bucket de Google Cloud."""
        path_bucket = self.model_dest+"/"+self.version_name
        nom_bucket = path_bucket.split('/')[2]
        path_file_bucket = path_bucket.split('gs://{}/'.format(nom_bucket))[1]+"/"+self.model_name + ".h5"

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(nom_bucket)
        blob = bucket.blob(path_file_bucket)

        blob.upload_from_filename(self.fname)

        logging.info('Modelo {} guardado en {}.'.format(
            self.model_name +'.h5',
            self.model_dest))

    def to_savedmodel(self):
        """
        Función que transforma el modelo de .h5 (Keras) a .pb(Tensorflow) y
        guarda ambos en Storage

        Returns:
            - res(string): ruta donde se guarda modelo transformado
        """
        model = load_model(self.fname)
        path = self.model_dest+"/"+self.version_name

        try:
            builder = SavedModelBuilder(path)
            signature = predict_signature_def(
                inputs={"inputs": model.input},
                outputs={"outputs": model.output})
            with K.get_session() as sess:
                builder.add_meta_graph_and_variables(
                    sess=sess,
                    tags=[tag_constants.SERVING],
                    signature_def_map={
                        'predict': signature})
                builder.save()
            res = "Modelo guardado en formato .pb en {}".format(path)
            logging.info(res)
        except AssertionError as exception:
            res = exception
            logging.error(exception)

        try:
            logging.info(f"Subiendo modelo en H5 a Storage: {path}")
            self.upload_blob()
        except AssertionError as exception:
            pass
            logging.error(f"Error al tratar de subir el modelo H5:{exception}')

        return res

    def create_model(self):
        """
        Función que crea una nueva instancia de modelo en el proyecto

        Returns:
            - new_model(string): La confirmación de que se instanció un nuevo modelo
        """
        command = """ gcloud ml-engine models create {} \
            --enable-logging \
            --regions us-central1 \
            --project {}""".format(self.model_name, self.project_id)
        new_model = os.system(command)

        if new_model > 0: #Si nos regresa un error el comando ejecutado
            logging.warning('Un modelo con el nombre %s ya existe', self.model_name)
            logging.warning('se intentará añadir una nueva versión')

        return new_model

    def create_version(self):
        """
        Función que crea una nueva versión del modelo entrenado

        Returns:
            - new_model(string): La descripción de la nueva versión del modelo
                                 desplegado en ML Engine, junto con el endpoint creado
        """
        #obtenemos el token que nos identifica en GCP
        batcmd = "gcloud auth print-access-token"
        token_curl = subprocess.check_output(batcmd, shell=True).decode("utf-8").split('\n')[0]

        endpoint = ""
        json_curl = {'name': self.version_name,
                     'pythonVersion': '3.5',
                     'deploymentUri': self.model_dest+'/'+self.version_name,
                     'runtimeVersion': '1.8'}
        headers_curl = {'Authorization': 'Bearer '+ token_curl}
        url = "https://ml.googleapis.com/v1/projects/{}/models/{}/versions".format(self.project_id, self.model_name)

        try:
            request = requests.post(url, json=json_curl, headers=headers_curl)
            endpoint = """https://ml.googleapis.com/v1/projects/{}/models/{}/versions/{}:predict
                """.format(self.project_id, self.model_name, self.version_name)
            mess = request.text + "\nEndpoint: " + endpoint
            logging.info("INFO: Endpoint creado: %s", mess)

        except requests.exceptions.RequestException as exception:
            mess = "Error: %s", request.text
            logging.error(mess)
            logging.error("Exception: %s", exception)

        return mess

    def deploy(self):
        """
        Función que despliega el modelo guardado en .h5 a endpoint en ML Engine

        Returns:
            - res(string): La descripción de la nueva versión del modelo desplegado en
                                 ML Engine, junto con el endpoint creado
        """
        logging.info("Convirtiendo modelo y guardando en Storage...")
        self.to_savedmodel()

        logging.info("Creando nuevo modelo en ML Engine...")
        self.create_model()

        logging.info("Creando nueva versión del modelo...")
        res = self.create_version()

        return res

#!/usr/bin/python3
# coding: utf-8
"""
Clase que hace deploy a ML Engine.
"""
import os
import logging
import subprocess
import requests
import tensorflow as tf
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.saved_model import tag_constants
import keras.backend as K
from keras.models import load_model


logger = logging.getLogger('deploy_mlengine')
logger.setLevel(logging.DEBUG)

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

    def to_savedmodel(self):
        """
        Función que transforma el modelo de .h5 (Keras) a .pb(Tensorflow) y lo
        guarda en Storage
        """
        model = load_model(self.fname)

        try:
            builder = SavedModelBuilder(self.model_dest+"/"+self.version_name)
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
            res = "Modelo guardado en formato .pb"
            logger.info(res)
        except AssertionError as exception:
            res = exception
            logger.error(exception)

        return res

    def create_model(self):
        """
        Función que crea una nueva instancia de modelo en el proyecto
        """
        command = """ gcloud ml-engine models create {} \
            --enable-logging \
            --regions us-central1 \
            --project {}""".format(self.model_name, self.project_id)
        new_model = os.system(command)

        if new_model > 0: #Si nos regresa un error el comando ejecutado
            logger.warning('Un modelo con el nombre %s ya existe, se intentará\
            añadir una nueva versión', self.model_name)

        return new_model

    def create_version(self):
        """
        Función que crea una nueva versión del modelo entrenado
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
            mess = request.text + "\n" + endpoint
            logger.info("INFO: Endpoint creado: %s", mess)

        except requests.exceptions.RequestException as exception:
            logger.error("Error: %s", request.text)
            logger.error("Exception: %s", exception)

        return request

    def deploy(self):
        """
        Función que despliega el modelo guardado en .h5 a endpoint en ML Engine
        """
        logger.info("Convirtiendo modelo y guardando en Storage...")
        self.to_savedmodel()

        logger.info("Creando nuevo modelo en ML Engine...")
        self.create_model()

        logger.info("Creando nueva versión del modelo...")
        res = self.create_version()

        return res

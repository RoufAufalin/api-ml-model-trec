import os
import uvicorn
import traceback
import tensorflow as tf
import tensorflow_text
import numpy as np

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response

interpreter = tf.lite.Interpreter(model_path='./converted_model.tflite')


app = FastAPI()


@app.get("/")
def index():
    return "Hello world from ML endpoint!"


class RequestText(BaseModel):
    text: str


@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:

        text = req.text
        print("Uploaded text:", text)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_data = np.array([text])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        result = interpreter.get_tensor(output_details[0]['index'])
        labels = ['Teknik Informatika, Sistem Informasi, Ilmu Komputer',
                  'Ekonomi, Akuntansi, Manajemen',
                  'Seni, Desain Komunikasi Visual, Desain Produk',
                  'Kedokteran, Kesehatan Masyarakat, Keperawatan']

        index = tf.argmax(result, axis=1).numpy()[0]
        return labels[index]

        return "Endpoint not implemented"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


port = os.environ.get("PORT", 4000)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
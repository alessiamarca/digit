from gradio.components import Image
from gradio.components import Label
from gradio import Interface
import numpy as np 

import joblib

model=joblib.load(filename="models/digit_pipeline.joblib")

input_image= Image(shape=(8,8), image_mode="L", invert_colors=True, source = "canvas", label="INPUT DIGIT")
output_image =Label(num_top_classes=10, label="MODEL PREDICTIONS")
title="Digit classifier with ML"
description="<center>This project is a demo or the class of TAG DS&AI master </center>"

def predict_image(image):
    labels = np.arange(start=0, stop =10, dtype=int).astype(str)
    flat_image= image.reshape(-1,64) #valori nell'array devono avere la stessa scala dell'originale da 255 a 16
    scaled_image = ((flat_image /255)*16).astype(np.uint16)
    #ottengo probabilità associate ad ogni etichetta
    probas =model.predict_proba(X=scaled_image)[0]
    #accoppio etichetta con probabilità
    result = {label: proba for label,proba in zip(labels,probas)}
    return result

interface =Interface(
    fn=predict_image,
    inputs=input_image,
    outputs=output_image,
    title=title,
    description= description,
)
interface.launch()
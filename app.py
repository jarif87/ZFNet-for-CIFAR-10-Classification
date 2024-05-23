import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.image import resize

# Load the trained model
model = load_model("trained_model_10.h5")

# CIFAR-10 labels
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the function to recognize images
def recognize_image(image):
    # Preprocess the image to fit the model input requirements
    img = keras_image.img_to_array(image)
    img = resize(img, (32, 32))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalizing if the model expects normalized input
    
    # Make predictions
    pred = model.predict(img)
    final_pred = np.argmax(pred, axis=1)
    
    # Create a dictionary mapping labels to their respective probabilities
    result = {label_names[i]: float(pred[0][i]) for i in range(len(label_names))}
    
    return result

# Define the input and output interfaces
image_input = gr.Image()
label_output = gr.Label(num_top_classes=5)

# Example images
examples = [
    'image_1.jpeg',
    'image_2.jpg',
    'image_4.jpeg',
    'image_5.jpg',
    'image_7.jpg',
    'image_8.jpeg'
]

# Create the Gradio interface
iface = gr.Interface(fn=recognize_image, inputs=image_input, outputs=label_output, examples=examples)

# Launch the interface
iface.launch(inline=False)

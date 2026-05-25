from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer 
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to caption a single image
def generate_caption(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Upload and caption
from google.colab import files
uploaded = files.upload()

for file_name in uploaded.keys():
    caption = generate_caption(file_name)
    img = Image.open(file_name)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()
    print("Generated Caption:", caption)

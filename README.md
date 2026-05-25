
#  Multimodal Image Captioning AI

An AI-powered Image Caption Generator built using Vision Transformers (ViT) and GPT-2. This project automatically generates meaningful captions for uploaded images using state-of-the-art deep learning models from the Transformers library.

---

##  Features

- Generate captions for real-world images
- Uses pre-trained VisionEncoderDecoder architecture
- Combines ViT (Vision Transformer) + GPT-2
- Supports GPU acceleration with PyTorch
- Easy to run in Google Colab
- Clean and beginner-friendly implementation

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Vision Transformer (ViT)
- GPT-2
- PIL (Python Imaging Library)
- Matplotlib
- Google Colab

---

##  Project Structure

```bash
├── image_captioning.py
├── sample_images/
├── outputs/
└── README.md
```

---

##  Installation

Install the required libraries before running the project:

```bash
pip install transformers torch pillow matplotlib
```

---

## How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/multimodal-image-captioning-ai.git
cd multimodal-image-captioning-ai
```

### Step 2: Run the Python File

```bash
python image_captioning.py
```

### Step 3: Upload an Image

The program will allow image upload and automatically generate captions.

---

##  Code Overview

### Load Pre-trained Model

```python
model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
```

### Generate Caption

```python
output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
```

---

##  Sample Output

| Input Image | Generated Caption |
|-------------|------------------|
| Dog Image | "a dog running through the grass" |
| Car Image | "a red sports car parked on the road" |

---

##  Model Used

This project uses the pre-trained model:

- `"nlpconnect/vit-gpt2-image-captioning"`

Architecture:
- **Encoder:** Vision Transformer (ViT)
- **Decoder:** GPT-2 Language Model

---

##  Applications

- Accessibility tools for visually impaired users
- Social media auto-captioning
- Image search engines
- Smart photo management systems
- AI-powered content generation

---

##  Future Improvements

- Real-time webcam captioning
- Support for multiple languages
- Fine-tuning on custom datasets
- Deploy as a web application
- Add voice output for captions

---

##  Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License.

---

##  Author

Developed with  using Deep Learning and Transformers.
````

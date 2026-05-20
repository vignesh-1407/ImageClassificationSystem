import os
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def get_image_path():
    if "--image" in sys.argv:
        index = sys.argv.index("--image") + 1
        if index < len(sys.argv):
            return sys.argv[index]

    if len(sys.argv) > 1:
        return sys.argv[1]

    return None

def get_prediction(image_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_model.h5")

    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}

    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    try:
        model = load_model(model_path)
    except Exception as e:
        return {"error": f"Error loading model: {e}"}

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((32, 32))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]

        top_indices = np.argsort(prediction)[::-1][:3]
        
        results = []
        for index in top_indices:
            results.append({
                "class_name": classes[index],
                "confidence": float(prediction[index] * 100)
            })
            
        return {"success": True, "predictions": results}
    except Exception as e:
        return {"error": f"Error processing image: {e}"}

def predict_image(image_path):
    result = get_prediction(image_path)
    
    if "error" in result:
        print(result["error"])
        return

    print("\nAI IMAGE CLASSIFICATION RESULT")
    print("=" * 40)
    print("Image:", image_path)
    print()

    for rank, pred in enumerate(result["predictions"], 1):
        print(f"{rank}. {pred['class_name']} - {pred['confidence']:.2f}%")

    print("=" * 40)

if __name__ == "__main__":
    image_path = get_image_path()

    if image_path is None:
        print("Please give image path")
        print("Example:")
        print("python predict.py --image C:\\Users\\DELL\\Downloads\\cat.WEBP")
    else:
        predict_image(image_path)
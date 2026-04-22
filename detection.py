

import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import cv2
import torch

# python -c "import torch; print(torch.cuda.is_available())"



# Cell
import random
from pathlib import Path
import cv2
from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # medium model (good balance)
class VisionAgent:
    def __init__(self):
        self.model = model

    def get_yolo_detections(self, img_path):
        """Run YOLO on an image and return a list of detected objects."""
        results = self.model(img_path, imgsz=960, conf=0.2)

        detected_objects = []

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                confidence = float(box.conf[0]) * 100

                detected_objects.append({
                    "label": label,
                    "confidence": round(confidence, 2)
                })

        return detected_objects

    def run(self, data):
        """
        'data' should be the image path to process.
        """
        img_path = data
        
        # Save shared path for the rest of the pipeline
        with open("shared_image.txt", "w") as f:
            f.write(img_path)

        detections = self.get_yolo_detections(img_path)
        
        # Return a neatly formatted dictionary of what was found
        return {
            "image_path": img_path,
            "detections": detections
        }

# Optional testing if run standalone
if __name__ == "__main__":
    import os
    dataset_path = Path("datasets/coco2017val/coco/images/val2017")
    image_files = list(dataset_path.glob("*.jpg"))

    # Select random image and save it for context-critic.ipynb
    if image_files:
        test_image_path = str(random.choice(image_files))
        print(f"Test Detection on Random Image: {test_image_path}")
        
        # Write the selected path to a shared file so context-critic.ipynb can read it
        with open("shared_image.txt", "w") as f:
            f.write(test_image_path)

        if "grounding_dino_detect" in globals():
            objects_dino = grounding_dino_detect(test_image_path)
        else:
            objects_dino = []
            print("[INFO] grounding_dino_detect is not defined; skipping Grounding DINO detections.")
        
            
        agent = VisionAgent()
        detections = agent.get_yolo_detections(test_image_path)
        print("Detected Objects:")
        for obj in detections:
            print(f"- {obj['label']} (Confidence: {obj['confidence']}%)")
        labels_yolo = [obj['label'] for obj in detections]
        labels_dino = [obj['label'] for obj in objects_dino] 

        objects = list(set(labels_yolo + labels_dino))
        
        # Display image with bounding boxes seamlessly drawn natively by YOLO
        results = agent.model(test_image_path)
        img_with_boxes = results[0].plot()
        
        cv2.imshow("YOLO Built-in Detection Test", img_with_boxes)
        print("\n[NOTE: Check your taskbar for the OpenCV window! Click on it and press ANY KEY to close it.]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("No test images found in dataset.")


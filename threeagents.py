class ContextAgent:
    def __init__(self):
        pass

    def run(self, data):
        print(f"[ContextAgent] Processing vision data -> Generating Captions")
        image_path = data.get("image_path")
        
        if not image_path or not os.path.exists(image_path):
            return {"error": "Invalid image path"}
            
        raw_image = Image.open(image_path).convert("RGB")
        
        # 1. First use BLIP to get a short caption
        blip_inputs = blip_processor(images=raw_image, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**blip_inputs, max_new_tokens=30)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        
        # Merge vision detections with context caption
        data["blip_caption"] = caption
        
        return data

# Cell
# %pip uninstall -y transformers tokenizers accelerate
# %pip install --upgrade transformers tokenizers accelerate huggingface_hub
# %pip install ultralytics transformers torch pillow requests bitsandbytes langchain-ollama

import torch
from PIL import Image
from pathlib import Path
from transformers import (AutoProcessor, AutoModel, BlipProcessor, BlipForConditionalGeneration,
                          InstructBlipProcessor, InstructBlipForConditionalGeneration)

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



# download('https://ultralytics.com/assets/coco2017val.zip', unzip=True, dir='datasets')


# Cell
clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Cell
import random
import os

# COCO dataset has 80 classes
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Convert to CLIP-compatible labels
labels = [f" A {cls}" for cls in coco_classes]

# Load images from local COCO dataset and select random shared image for all models
dataset_path = Path("datasets\\coco2017val\\coco\\images\\val2017")
# dataset_path = Path("datasets\\coco2017val\\coco\\images\\val2017")
image_files = sorted(dataset_path.glob("*.jpg"))

# Commenting out the immediate image selection so that it does not run at module import time.
# That logic should eventually be encapsulated inside the class's `run(...)` method.
selected_image_path = None
image = None
shared_file = "shared_image.txt"

'''
# Prioritize loading the image selected by detection.ipynb
if os.path.exists(shared_file):
    with open(shared_file, "r") as f:
        shared_path = f.read().strip()
    selected_image_path = Path(shared_path)
    if selected_image_path.exists():
        print(f"Loaded shared image from detection.ipynb: {selected_image_path.name}")
        image = Image.open(selected_image_path).convert("RGB")
    else:
        print(f"Shared image path {shared_path} does not exist.")
        selected_image_path = None
elif image_files:
    selected_image_path = random.choice(image_files)
    image = Image.open(selected_image_path).convert("RGB")
    print(f"No shared image found. Randomly selected image: {selected_image_path.name}")
else:
    print("No images found in dataset")
'''

# Cell
# Use image from local dataset
# if image:
#     inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
#     outputs = clip_model(**inputs)
#     print("Processing complete")
# else:
#     print("No image available to process")

# Cell
# logits_per_image = outputs.logits_per_image
# probs = logits_per_image.softmax(dim=1)

# Get all probabilities and sort them
# all_probs = probs[0].detach().float().cpu().numpy()
# sorted_indices = all_probs.argsort()[::-1]  

threshold = 0.015
scene_objects = []

# print("Scene: Objects Detected\n")


# for idx in sorted_indices:
#     prob_score = all_probs[idx]
    
#     if prob_score >= threshold:
#         object_name = labels[idx].strip()
#         confidence_pct = prob_score * 100
#         scene_objects.append((object_name, confidence_pct))
#         print(f"{object_name:30s} | Confidence: {confidence_pct:6.2f}%")



class LanguageAgent:
    def __init__(self):
        pass

    def run(self, data):
        print(f"[LanguageAgent] Describing image in detailed structure...")
        img_path = data.get("image_path")
        blip_caption = data.get("blip_caption", "No base caption available.")
        
        if not img_path or "error" in data:
            return data
            
        llava_response = describe_image(img_path, blip_caption, model_name="llava")
        
        # Add LLAVA detailed description to the output
        data["llava_description"] = llava_response
        return data

# Cell
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

if selected_image_path is not None and selected_image_path.exists():
    raw_image = Image.open(selected_image_path).convert("RGB")
    blip_inputs = blip_processor(images=raw_image, return_tensors="pt")

    with torch.no_grad():
        out = blip_model.generate(**blip_inputs, max_new_tokens=30)

    caption = blip_processor.decode(out[0], skip_special_tokens=True)
else:
    caption = "No image available for BLIP captioning."

# Cell
# from IPython.display import display

# print(f"Caption: {caption}")


# display(raw_image)

# Cell
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def describe_image(image_path, base_caption, model_name="llava"):
    """Use a local Vision LLM via Ollama to describe an image in detailed structure, guided by a short caption."""
    
    prompt = f"""You are an expert visual scene describer. 
I am providing you with an image and a short preliminary caption: "{base_caption}"
Look at the provided image and generate an extremely detailed description expanding upon the caption.
You MUST start with a new line after finishing every sentence.
You MUST structure your response EXACTLY using the following headings :

Objects and Items:
[Provide detailed description here]

Spatial Layout:
[Provide detailed description here.]

Lighting and Atmosphere:
[Provide detailed description here.]

People and Interactions:
[Provide detailed description here. If no people, state so.]

Visual Elements:
[Provide detailed description here.]

Context and Setting:
[Provide detailed description here.]
"""
 
    llm = ChatOllama(model=model_name, temperature=0.7)
    # print(f"Describing image: '{image_path}' using {model_name}...\n")
    
    try:
        image_b64 = image_to_base64(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ]
        )
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Error connecting to Ollama: {e}\nPlease ensure Ollama is running locally and the vision model (e.g., '{model_name}') is installed."

# if selected_image_path:
#     # Pass both the image path and the short 'caption' generated by BLIP earlier
#     detailed_notes = describe_image(selected_image_path, caption, model_name="llava")
#     print(detailed_notes)
# else:
#     print("No image selected.")
# detailed_notes = "No image selected"

class CriticAgent:
    def __init__(self):
        pass
        
    def run(self, data):
        print("[CriticAgent] Evaluating hallucination and applying logic...")
        
        objects = data.get("detections", [])
        detailed_notes = data.get("llava_description", "")
        
        objects_str = "\n".join([f"- {obj['label']} (Confidence: {obj['confidence']:.2f}%)" for obj in objects])
        
        # Part 1: Critic
        critic_chained = criticprompt | criticllm
        critic_response = critic_chained.invoke({
            "objects": objects_str,
            "description": detailed_notes
        }).content
        
        # Part 2: Reflector
        reflector_chained = reflectorprompt | reflectorllm
        reflector_response = reflector_chained.invoke({
            "prompt": detailed_notes,
            "criticprompt": critic_response
        }).content
        
        data["critic_evaluation"] = critic_response
        data["final_reflection"] = reflector_response
        
        # Update memory
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            frame_summary = f"[Time: {timestamp}] {reflector_response[:200]}..." 
            memory_agent.update(frame_summary)
        except Exception as e:
            data["memory_error"] = str(e)

        return data

# Cell
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


criticllm = ChatOllama(
    model="llama3.2", 
    temperature=0.2,
)
criticprompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert computer vision critic and image evaluator. Your job is to analyze the outputs of different vision models and provide a final, highly accurate evaluation of the scene."),
    ("human", """
    Please evaluate the generated image description based on the objects detected in the scene. 
    
    Detected Objects : 
    {objects}
    
    Generated Description : 
    "{description}"
    
    Task:
    1. Critique the description. Does it make sense given the highly confident detected objects? Are there any hallucinations?
    2. Provide a refined, final evaluation and suggestion for the image description.
    """)
])


# objects_str = "\n".join([f"- {obj[0]} (Confidence: {obj[1]:.2f}%)" for obj in scene_objects])


# criticchain = criticprompt | criticllm

# print("Critic Agent : - \n")
# response = criticchain.invoke({
#     "objects": objects_str,
#     "description": detailed_notes
# })

# print(response.content)

# Cell
reflectorllm = ChatOllama(
    model="llama3.2", 
    temperature=0.2,
)
reflectorprompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert computer vision critic and image evaluator. Your job is to apply the suggestions from the previous evaluation."),
    ("human", """
    Please apply the critique and suggestions from the previous evaluation to generate a final, highly accurate description of the scene.
    
    Original Description: "{prompt}"
    
    Critic's Evaluation: "{criticprompt}"
    
    To further improve the generated description, please ensure the following:
    * Use more descriptive language to paint a vivid picture of the scene.
    * Avoid vague phrases like "a person in the background" and instead specify the person's role or interaction with the scene.
    * Consider alternative captions that take into account the context and relationships between the detected objects.
    
    Task:
    1. Apply the critique and suggestions from the previous evaluation.
    2. Provide a refined, final description of the image that combines both inputs logically.
    """)
])

# reflectorchain = reflectorprompt | reflectorllm

# print("Reflector Agent : - \n")
# reflector_response = reflectorchain.invoke({
#     "prompt": detailed_notes,
#     "criticprompt": response.content
# })

# print(reflector_response.content)



# Cell
# Connect Memory Agent 
# from memory import MemoryAgent

# if 'memory_agent' not in globals():
#     memory_agent = MemoryAgent()
    

# # Update the memory with the final reflection
# import datetime
# timestamp = datetime.datetime.now().strftime("%H:%M:%S")

# frame_summary = f"[Time: {timestamp}] {reflector_response.content[:200]}..." 
# memory_agent.update(frame_summary)

# print(f"\n Updated Memory Agent. Current Memory Size: {len(memory_agent.history)} frames.")
# print("\n Memory Summary ")
# print(memory_agent.summarize())

# Connect Communication Agent
# from communication import CommunicationAgent

# if 'comm_agent' not in globals():
#     comm_agent = CommunicationAgent()

# # Create a mock agent for context to receive messages
# class ContextAgent:
#     def run(self, data):
#         return f"Context received: {data}"

# # Register the context agent
# agent_dict = {
#     "context": ContextAgent()
# }
# comm_agent.register_agents(agent_dict)

# # Send the frame summary to the context agent
# message = comm_agent.send(
#     sender="vision",
#     receiver="context",
#     data=frame_summary,
#     msg_type="data"
# )

# # Route the message
# response = comm_agent.route(message)

# print(f"\n Dispatched Message via Communication Agent:")
# print(f"Message ID: {message['id']}")
# print(f"Response: {response}")
# print(f"Current Logs: {len(comm_agent.log)}")


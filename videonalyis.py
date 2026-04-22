import cv2 
import os
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration,BitsAndBytesConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from PIL import Image
import httpx
from io import BytesIO
import av
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration



processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b", device_map="auto" ,torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT



text={}
def extract_all_frame(video_path,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("error:could not open video")
    print("Video opened succesfully")
    
    frame_count=0
    short_frame_count=0
    original_fps=30
    new_fps=2
    interval=original_fps//new_fps
    while True:

        ret,frame=cap.read()
        if ret == False:
            print("end of video reached")
            break
        
        frame_count+=1
        filename=os.path.join(output_folder,f"frame_{frame_count:04d}.jpg")
        if frame_count % interval==0:
            raw_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            prompt = "give a short description regarding what is happening in the image including only visible details"
            inputs = processor(images=raw_image,text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

            generated_ids = model.generate(**inputs,max_new_tokens=300,do_sample=False,num_beams=3)
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print("decsription of frame",frame_count," ",generated_text)
            text[frame_count]=generated_text

            cv2.imwrite(filename,frame)
            short_frame_count+=1
        
    cap.release()
    print("Frame count=",frame_count)


extract_all_frame("path to your video","short_extracted_frames")#in the first argument give the path to yoyr video



highest=0#frame which will egt highest mmr score
def mmr_summarize(text,frames_size,lambda_param=0.7):
    

    # 2. TF-IDF sentence embeddings
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_embeddings = vectorizer.fit_transform(text)

    # 3. Document embedding (mean of description vectors)
    doc_embedding = np.mean(sentence_embeddings.toarray(), axis=0).reshape(1, -1)

    # 4. Similarity matrices
    sim_to_doc = cosine_similarity(sentence_embeddings, doc_embedding).flatten()
    sim_between_sentences = cosine_similarity(sentence_embeddings)

    # 5. MMR selection
    selected = []
    candidate_indices = list(range(len(text)))

    # Step 1: pick most relevant frame
    first_idx = np.argmax(sim_to_doc)
    selected.append(first_idx)
    candidate_indices.remove(first_idx)

    # Step 2: pick remaining frames using MMR
    for _ in range(frames_size):
        mmr_scores = []

        for idx in candidate_indices:
            relevance = sim_to_doc[idx]
            diversity = max(sim_between_sentences[idx][s] for s in selected)
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((mmr, idx))

        selected_idx = max(mmr_scores)[1]
        selected.append(selected_idx)
        candidate_indices.remove(selected_idx)

    # 6. Return most important frames in original order
    print("indices of most relevant frames ",selected)
    highest=selected[0]
    
    selected = sorted(selected)
    

    return selected

indices=mmr_summarize(text.values(),9)
print(indices)

indices=[0]+indices
k=0
for i in indices:
    indices[k]=(i+1)*15-1
    k+=1
print(indices)#indices of most relevant frames to be given to video llava for video description and temporal reasoning


#starting of video llava for decoding frames and generating video description
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",torch_dtype=torch.float16,device_map="auto")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

prompt = "<video>\Describe the video in details explaining key events like the collision,interactions with temporal reasoning."
#prompt="<video>\at the end how the collision with tyre impacted the blue truck "
video_path = "/teamspace/uploads/VIDEO.mp4"
container = av.open(video_path)
total_frames = container.streams.video[0].frames
clip = read_video_pyav(container, indices)
print(indices)

inputs = processor(text=prompt, videos=clip, return_tensors="pt")
inputs={k:v. to ("cuda") for k,v in inputs.items()}

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=400,do_sample=False,num_beams=5)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]





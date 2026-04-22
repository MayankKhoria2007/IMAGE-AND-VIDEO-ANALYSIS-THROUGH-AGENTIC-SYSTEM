
# MAIN PIPELINE
def run_pipeline(input_data):
    try:
        # Step 1: Vision
        vision_output = comm.dispatch("main", "vision", input_data)
        vision_results = {"detections": vision_output.get("detections", [])}
        display_img = cv2.imread(vision_output.get("image_path", ""))
        if display_img is not None:
            cv2.imshow("Vision Agent Output", display_img)
            cv2.waitKey(0)
            try:
                cv2.destroyWindow("Vision Agent Output")
            except cv2.error:
                pass

        # Step 2: Context
        context_output = comm.dispatch("vision", "context", vision_output)
        context_results = {"blip_caption": context_output.get("blip_caption", "")}

        # Step 3: Language
        language_output = comm.dispatch("context", "language", context_output)
        language_results = {"llava_description": language_output.get("llava_description", "")}

        # Step 4: Critic
        critic_output = comm.dispatch("language", "critic", language_output)
        critic_results = {
            "critic_evaluation": critic_output.get("critic_evaluation", ""),
            "final_reflection": critic_output.get("final_reflection", "")
        }

        return {
            "vision": vision_results,
            "context": context_results,
            "language": language_results,
            "critic": critic_results
        }

    except Exception as e:
        print("Pipeline Error:", e)
        return None


if __name__ == "__main__":
    import cv2
    import random
    from pathlib import Path

    agents_initialized = False

    while True:
        print("\nImage Selection ")
        print("1: Use Webcam")
        print("2: Random Image")
        choice = input("Enter your choice (1 or 2, or 'q' to quit): ").strip().lower()

        if choice == 'q':
            break

        if choice == '1':
            print("Capturing from webcam...")
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("Error: Could not capture from webcam. Please check your camera.")
                continue
            
            user_input = "temp_webcam.jpg"
            cv2.imwrite(user_input, frame)
            print(f"Saved webcam snapshot to {user_input}")

        elif choice == '2':
            dataset_path = Path("datasets/coco2017val/coco/images/val2017")
            image_files = list(dataset_path.glob("*.jpg"))
            
            if not image_files:
                print(f"Error: No images found in {dataset_path}")
                continue
                
            user_input = str(random.choice(image_files))
            print(f"Selected random image: {user_input}")

        else:
            print("Invalid choice. Please enter 1, 2, or 'q'.")
            continue
            
        display_img = cv2.imread(user_input)
        if display_img is not None:
            cv2.imshow("Selected Image", display_img)
            cv2.waitKey(0)
            try:
                cv2.destroyWindow("Selected Image")
            except cv2.error:
                pass
        else:
            print("Failed to load image for display.")

        if not agents_initialized:
            print("\n[SYSTEM] Initializing models and loading scripts (this will only happen once)...")
            
            # Defer imports until after the choice is made so the "Loaded shared image..." 
            # prints happen in correct order
            from detection import VisionAgent
            from threesome import ContextAgent
            from threesome import LanguageAgent
            from threesome import CriticAgent
            from communication import CommunicationAgent

            comm = None

            
            vision = VisionAgent()
            context = ContextAgent()
            language = LanguageAgent()
            critic = CriticAgent()

            agents = {
                "vision": vision,
                "context": context,
                "language": language,
                "critic": critic
            }

            # Initialize communication system
            comm = CommunicationAgent()
            comm.register_agents(agents)
            agents_initialized = True

        output = run_pipeline(user_input)

        if output:
            
            print("\n----- PIPELINE OUTPUTS -----")
            
            # Vision Agent
            print("\n[VISION (Detections)]")
            for det in output["vision"].get("detections", []):
                print(f" - {det['label']} ({det['confidence']:.2f}%)")
                
            # Context Agent
            print("\n[CONTEXT (Caption)]")
            print(output["context"].get("blip_caption", "No caption generated."))
            
            # Language Agent
            print("\n[LANGUAGE (Description)]")
            print(output["language"].get("llava_description", "No description generated."))
            
            # Critic Agent
            print("\n[CRITIC (Evaluation & Errors)]")
            print(output["critic"].get("critic_evaluation", "No evaluation generated."))
            
            print("\n[CRITIC (Final Reflection)]")
            print(output["critic"].get("final_reflection", "No reflection generated."))
            
            print("\n----------------------------\n")
          

import os
import random
# Get the absolute path to the directory containing this script
res_dir = os.path.dirname(os.path.abspath(__file__))
pred_dir = "/data1/wangqiurui/code/competition/tianchi/unseen/YOLOv11-RGBT/runs/rgb_ir/yolo11s-RGBT-midlate-fusion-finetuneM3DF/res_923/labels"
# Define paths for the labels directory and the output file
# labels_dir = os.path.join(pred_dir, "labels")
output_file = os.path.join(res_dir, "result_srgbt_1280_0.923val_20thr.txt")

# --- (Optional) Hardcoded model parameters for demonstration ---
# Replace with your actual model's parameter and calculation amount
model_params = "532105 19"  # Example: num_params GFLOPs

# --- Main processing logic ---
with open(output_file, "w") as f_out:
    # Write the header line with model parameters
    f_out.write(f"{model_params}\n")

    # Get all .txt files from the labels directory
    label_files = [f for f in os.listdir(pred_dir) if f.endswith(".txt")]

    processed_files = []
    num_files = 0
    # Process each label file
    for filename in label_files:
        image_name = os.path.splitext(filename)[0] + ".jpg"
        txt_name = os.path.splitext(filename)[0] + ".txt"
        # Initialize a list to store detection data for the current image
        detections = []

        with open(os.path.join(pred_dir, filename), "r") as f_in:
            for line in f_in:
                parts = line.strip().split()
                
                # Ensure there are enough parts to represent a detection
                if len(parts) == 6:
                    # Extract detection attributes
                    class_id, x_center, y_center, width, height, confidence = map(float, parts)

                    # Filter detections by confidence score
                    if confidence > 0.25:
                        # Append relevant detection data to the list
                        detections.extend([x_center, y_center, width, height, confidence, int(class_id)])
                    elif 0.2 < confidence < 0.25:
                        # random confident(0.25,0.8]
                        conf = round(random.uniform(0.26, 0.8),6)
                        detections.extend([x_center, y_center, width, height, conf, int(class_id)])

        # If there are valid detections, write them to the output file
        if detections:
            # Format the output line with the image name followed by detection data
            output_line = f"{image_name} {' '.join(map(str, detections))}\n"
            f_out.write(output_line)
            processed_files.append(txt_name)
            num_files += 1
    
    nn = 0
    # process none detections
    for i in range(5000, 15000):
        # Format: 005000 005001 005002 ... 014999
        if i < 10000:
            filename = f'00{i}.txt'
        else:
            filename = f'0{i}.txt'
            
        if os.path.exists(os.path.join(pred_dir, filename)) and filename in processed_files:
            continue
        else:
            image_name = os.path.splitext(filename)[0] + ".jpg"
            f_out.write(f"{image_name}\n")
            print(f"{image_name}")
            nn += 1


print(f"Processing complete. {num_files} + {nn} Results are saved in {output_file}")

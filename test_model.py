import glob
import os
import cv2
from ultralytics import YOLO

print("Searching for the latest YOLO model...")
list_of_files = glob.glob('runs/**/weights/best.pt', recursive=True)

if not list_of_files:
    print(" No trained model found! Please train the model first.")
    exit()


latest_model_path = max(list_of_files, key=os.path.getctime)
print(f" Loading custom trained model from {latest_model_path}...\n")
model = YOLO(latest_model_path)

print(" Running predictions on the test dataset...")
print("This will process the images and open windows showing the detected objects.")
print("Press any key on the image window to proceed to the next one, or 'q' to quit.\n")


test_images = glob.glob('dataset/test/*.jpg') + glob.glob('dataset/test/*.png')

if not test_images:
    print(" No images found in dataset/test/!")
    exit()

for img_path in test_images:
    # Predict
    results = model.predict(source=img_path, save=True, conf=0.5)
    
   
    if len(results) > 0:
        res_plot = results[0].plot()  # 
        
        cv2.imshow("Testing YOLO Model", res_plot)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("\n Stopping test early.")
            break

cv2.destroyAllWindows()
print("\n Testing complete!")
print("You can also view all the permanently saved result images in the 'runs/detect/predict' folder.")

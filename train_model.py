import json
import os
import yaml
import argparse
import random
from ultralytics import YOLO

def convert_coco_to_yolo(sample_ratio=1.0, max_samples=0):
    dataset_dir = "dataset"
    splits = ["train", "valid", "test"]

    
    yaml_dict = {
        "train": os.path.join(os.path.abspath(dataset_dir), "train.txt"),
        "val": os.path.join(os.path.abspath(dataset_dir), "valid.txt"),
        "test": os.path.join(os.path.abspath(dataset_dir), "test.txt"),
        "nc": 0,
        "names": []
    }

    def get_class_id(category_id, categories):
        for i, cat in enumerate(categories):
            if cat['id'] == category_id:
                return i
        return 0

    print("Converting COCO JSON annotations to YOLO txt format & filtering dataset...")
    for split in splits:
        json_path = os.path.join(dataset_dir, split, "_annotations.coco.json")
        if not os.path.exists(json_path):
            print(f"File {json_path} not found, skipping {split} split.")
            yaml_dict[split if split != "valid" else "val"] = os.path.join(os.path.abspath(dataset_dir), split)
            continue
        
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
            
        categories = coco.get("categories", [])
        if len(yaml_dict["names"]) == 0:
            yaml_dict["names"] = [cat["name"] for cat in categories]
            yaml_dict["nc"] = len(categories)
            
        images = coco.get("images", [])
        
        
        random.seed(42) 
        random.shuffle(images)
        
        if max_samples > 0:
            images = images[:max_samples]
        elif sample_ratio < 1.0:
            num_samples = max(1, int(len(images) * sample_ratio))
            images = images[:num_samples]
            
        print(f"Split '{split}': Randomly selected {len(images)} images (out of {len(coco.get('images', []))})")
        
        
        image_id_to_file = {img["id"]: img.get("file_name", img.get("filename")) for img in images}
        image_id_to_size = {img["id"]: (img.get("width", 640), img.get("height", 640)) for img in images}
        
        labels = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in image_id_to_file:
                continue
                
            cat_id = ann["category_id"]
            cls_id = get_class_id(cat_id, categories)
            
            bbox = ann["bbox"] 
            img_w, img_h = image_id_to_size[img_id]
            
      
            x_c = (bbox[0] + bbox[2] / 2) / img_w
            y_c = (bbox[1] + bbox[3] / 2) / img_h
            w = bbox[2] / img_w
            h = bbox[3] / img_h
            
           
            x_c = max(min(x_c, 1.0), 0.0)
            y_c = max(min(y_c, 1.0), 0.0)
            w = max(min(w, 1.0), 0.0)
            h = max(min(h, 1.0), 0.0)
            
            file_name = image_id_to_file[img_id]
            base_name = os.path.splitext(file_name)[0]
            
            txt_path = os.path.join(dataset_dir, split, f"{base_name}.txt")
            line = f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
            
            if txt_path not in labels:
                labels[txt_path] = []
            labels[txt_path].append(line)
            
        
        for txt_path, lines in labels.items():
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
                
        
        img_list_path = os.path.join(os.path.abspath(dataset_dir), f"{split}.txt")
        with open(img_list_path, "w", encoding="utf-8") as f:
            for img in images:
                file_name = img.get("file_name", img.get("filename"))
                abs_img_path = os.path.join(os.path.abspath(dataset_dir), split, file_name)
                f.write(abs_img_path + "\n")
                
        yaml_key = split if split != "valid" else "val"
        yaml_dict[yaml_key] = img_list_path
                
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)
    
    print(f"Created YOLO data.yaml with {yaml_dict['nc']} classes at {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for AutoBatch)")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--sample", type=float, default=0.25, help="Fraction of the dataset to use (e.g., 0.25 for 25%). Set to 1.0 for whole dataset.")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit max images per split if > 0")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement). 0 to disable.")
    args = parser.parse_args()

    yaml_path = convert_coco_to_yolo(sample_ratio=args.sample, max_samples=args.max_samples)

    print(f"\nStarting YOLO training for {args.epochs} epochs using {int(args.sample * 100)}% of the dataset...")
    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: No GPU detected! Training on CPU takes a long time.")
    else:
        print(f"GPU detected ({torch.cuda.get_device_name(0)}). Training will be fast.")

    model = YOLO("yolov8n.pt")  
    
    
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        cache=True,            
        amp=True,              
        patience=args.patience, 
        project="runs/detect", 
        name="train",          
        exist_ok=True         
    )
    
    print("\n Training is complete!")
    print("The best custom model is saved to runs/detect/train/weights/best.pt")

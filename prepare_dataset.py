import os
import argparse
import shutil
import glob
import random
from pathlib import Path

def create_dataset_structure(base_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Create directory structure for YOLO dataset
    
    Args:
        base_dir: Base directory for the dataset
        split_ratio: Train/val/test split ratio (tuple of 3 values summing to 1)
    """
    print(f"Creating dataset structure in {base_dir}")
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(base_dir, split, subdir), exist_ok=True)
    
    print("Dataset structure created successfully")
    print(f"  - Train: {os.path.join(base_dir, 'train')}")
    print(f"  - Validation: {os.path.join(base_dir, 'val')}")
    print(f"  - Test: {os.path.join(base_dir, 'test')}")

def split_dataset(images_dir, labels_dir, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Split a dataset into train/val/test sets
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        output_dir: Output directory for the dataset
        split_ratio: Train/val/test split ratio (tuple of 3 values summing to 1)
    """
    print(f"Splitting dataset from {images_dir} and {labels_dir}")
    
    # Verify split ratio
    if sum(split_ratio) != 1.0:
        print("Warning: Split ratio doesn't sum to 1. Normalizing...")
        total = sum(split_ratio)
        split_ratio = tuple(x/total for x in split_ratio)
    
    # Create dataset structure
    create_dataset_structure(output_dir, split_ratio)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    n_train = int(len(image_files) * split_ratio[0])
    n_val = int(len(image_files) * split_ratio[1])
    
    # Split datasets
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Copy files to respective directories
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        print(f"Copying {len(files)} files to {split_name} set")
        
        for img_path in files:
            # Get filename without extension
            img_filename = os.path.basename(img_path)
            base_name = os.path.splitext(img_filename)[0]
            
            # Copy image
            dst_img_path = os.path.join(output_dir, split_name, 'images', img_filename)
            shutil.copy2(img_path, dst_img_path)
            
            # Copy corresponding label if it exists
            label_file = os.path.join(labels_dir, f"{base_name}.txt")
            if os.path.exists(label_file):
                dst_label_path = os.path.join(output_dir, split_name, 'labels', f"{base_name}.txt")
                shutil.copy2(label_file, dst_label_path)
            else:
                print(f"Warning: No label file found for {img_filename}")
    
    # Print summary
    print("\nDataset split complete:")
    print(f"  - Train set: {len(train_files)} images")
    print(f"  - Validation set: {len(val_files)} images")
    print(f"  - Test set: {len(test_files)} images")

def convert_labelimg_to_yolo(input_dir, output_dir, class_mapping=None):
    """
    Convert LabelImg XML annotations to YOLO format
    
    Args:
        input_dir: Directory containing XML annotations
        output_dir: Output directory for YOLO labels
        class_mapping: Dictionary mapping class names to indices (optional)
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        print("Error: XML module not available. Cannot convert LabelImg annotations.")
        return
    
    print(f"Converting annotations from {input_dir} to YOLO format")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default class mapping (assuming 'hand' is the only class)
    if class_mapping is None:
        class_mapping = {'hand': 0}
    
    # Get all XML files
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # Prepare YOLO format file
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            out_file = os.path.join(output_dir, f"{base_name}.txt")
            
            with open(out_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    
                    # Skip if class not in mapping
                    if class_name not in class_mapping:
                        print(f"Warning: Class '{class_name}' not in mapping, skipping")
                        continue
                    
                    class_idx = class_mapping[class_name]
                    
                    # Get bounding box
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # Convert to YOLO format (center_x, center_y, width, height)
                    center_x = (xmin + xmax) / (2 * width)
                    center_y = (ymin + ymax) / (2 * height)
                    bbox_width = (xmax - xmin) / width
                    bbox_height = (ymax - ymin) / height
                    
                    # Write to file
                    f.write(f"{class_idx} {center_x} {center_y} {bbox_width} {bbox_height}\n")
        
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
    
    print(f"Converted {len(xml_files)} XML files to YOLO format")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # create-structure command
    create_parser = subparsers.add_parser('create-structure', help='Create dataset directory structure')
    create_parser.add_argument('--dir', type=str, required=True, help='Base directory for the dataset')
    
    # split-dataset command
    split_parser = subparsers.add_parser('split-dataset', help='Split dataset into train/val/test sets')
    split_parser.add_argument('--images', type=str, required=True, help='Directory containing images')
    split_parser.add_argument('--labels', type=str, required=True, help='Directory containing labels')
    split_parser.add_argument('--output', type=str, required=True, help='Output directory for the dataset')
    split_parser.add_argument('--split', type=str, default='0.7,0.15,0.15', 
                             help='Train/val/test split ratio (comma-separated)')
    
    # convert-labels command
    convert_parser = subparsers.add_parser('convert-labels', help='Convert LabelImg XML annotations to YOLO format')
    convert_parser.add_argument('--input', type=str, required=True, help='Directory containing XML annotations')
    convert_parser.add_argument('--output', type=str, required=True, help='Output directory for YOLO labels')
    
    args = parser.parse_args()
    
    if args.command == 'create-structure':
        create_dataset_structure(args.dir)
        
    elif args.command == 'split-dataset':
        # Parse split ratio
        split_ratio = tuple(float(x) for x in args.split.split(','))
        split_dataset(args.images, args.labels, args.output, split_ratio)
        
    elif args.command == 'convert-labels':
        convert_labelimg_to_yolo(args.input, args.output)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

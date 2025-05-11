import os
import argparse
import yaml
from ultralytics import YOLO

def train_custom_model(data_yaml_path, epochs=100, patience=30, imgsz=640, batch_size=16, weights='yolov8n.pt'):
    """
    Train a custom YOLOv8 model on hand data
    
    Args:
        data_yaml_path: Path to YAML file containing dataset information
        epochs: Number of training epochs
        patience: Patience for early stopping
        imgsz: Image size for training
        batch_size: Batch size for training
        weights: Initial weights (pretrained model)
    """
    print(f"Training custom YOLOv8 model with the following parameters:")
    print(f"  - Data: {data_yaml_path}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Initial weights: {weights}")
    
    # Load a pretrained model
    model = YOLO(weights)
    
    # Train the model on your custom dataset
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch_size,
        name='hand_detection_model'  # Name for the saved model and logs
    )
    
    # Validate the model
    metrics = model.val()
    
    print("Training complete!")
    print(f"Model saved to: {os.path.join('runs', 'detect', 'hand_detection_model')}")
    
    return results

def create_hand_dataset_yaml(dataset_dir, output_yaml='data.yaml'):
    """
    Create a YAML file for the hand detection dataset
    
    Args:
        dataset_dir: Path to the dataset directory (with train, val subdirectories)
        output_yaml: Path to save the YAML file
    """
    # Create the YAML dictionary
    data = {
        'path': dataset_dir,
        'train': os.path.join(dataset_dir, 'train', 'images'),
        'val': os.path.join(dataset_dir, 'val', 'images'),
        'test': os.path.join(dataset_dir, 'test', 'images') if os.path.exists(os.path.join(dataset_dir, 'test')) else '',
        'names': {
            0: 'hand',
        },
        'nc': 1  # Number of classes
    }
    
    # Write the YAML file
    with open(output_yaml, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created dataset YAML file: {output_yaml}")
    
    return output_yaml

def main():
    parser = argparse.ArgumentParser(description='Train a custom YOLOv8 model for hand detection')
    
    # Dataset arguments
    parser.add_argument('--dataset-dir', type=str, required=True, 
                        help='Path to the dataset directory (with train, val subdirectories)')
    parser.add_argument('--data-yaml', type=str, default='data.yaml',
                        help='Path to save the dataset YAML file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', 
                        help='Initial weights (pretrained model)')
    
    args = parser.parse_args()
    
    # Create the dataset YAML file
    yaml_path = create_hand_dataset_yaml(args.dataset_dir, args.data_yaml)
    
    # Train the model
    train_custom_model(
        data_yaml_path=yaml_path,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch_size=args.batch,
        weights=args.weights
    )

if __name__ == "__main__":
    main()

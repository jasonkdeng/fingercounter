import os
import argparse
import requests
from PIL import Image
from io import BytesIO

def download_sample_images(output_dir="sample_images"):
    """
    Download sample hand images for testing the finger counter
    
    Args:
        output_dir: Directory to save the images
    """
    print(f"Downloading sample hand images to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List of sample hand images (these are placeholder URLs - replace with actual image URLs)
    # For a real implementation, you'd need to use actual URLs of hand images
    sample_images = [
        {"url": "https://i.stack.imgur.com/ANJlS.jpg", "name": "hand_five_fingers.jpg"},
        {"url": "https://i.stack.imgur.com/cQJpK.jpg", "name": "hand_peace_sign.jpg"},
        {"url": "https://i.stack.imgur.com/xMF8A.jpg", "name": "hand_pointing.jpg"},
        {"url": "https://us.123rf.com/450wm/artursz/artursz1909/artursz190900049/129504554-cropped-view-of-man-showing-three-fingers-isolated-on-white.jpg", "name": "hand_three_fingers.jpg"},
        {"url": "https://thumbs.dreamstime.com/b/male-hand-showing-four-fingers-isolated-white-background-male-hand-showing-four-fingers-184706345.jpg", "name": "hand_four_fingers.jpg"}
    ]
    
    # Download each image
    downloaded_count = 0
    for img_info in sample_images:
        try:
            print(f"Downloading {img_info['name']}...")
            response = requests.get(img_info['url'], stream=True)
            
            if response.status_code == 200:
                # Save the image
                img_path = os.path.join(output_dir, img_info['name'])
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify the image can be opened
                try:
                    img = Image.open(img_path)
                    img.verify()
                    print(f"Successfully downloaded and verified {img_info['name']}")
                    downloaded_count += 1
                except:
                    print(f"Downloaded file {img_info['name']} is not a valid image. Removing...")
                    os.remove(img_path)
            else:
                print(f"Failed to download {img_info['name']} (Status code: {response.status_code})")
                
        except Exception as e:
            print(f"Error downloading {img_info['name']}: {e}")
    
    print(f"Downloaded {downloaded_count} sample images to {output_dir}")
    
    # If no images were downloaded, provide guidance
    if downloaded_count == 0:
        print("No images were downloaded. The provided URLs may be invalid or inaccessible.")
        print("You can manually add sample hand images to the sample_images directory.")

def main():
    parser = argparse.ArgumentParser(description='Download sample hand images for testing')
    parser.add_argument('--output-dir', type=str, default='sample_images', 
                        help='Directory to save the sample images')
    args = parser.parse_args()
    
    download_sample_images(args.output_dir)

if __name__ == "__main__":
    main()

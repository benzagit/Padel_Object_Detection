import os
import xml.etree.ElementTree as ET
import cv2
from pathlib import Path

def xml_to_mot(xml_dir, image_dir, output_dir, img_size=(512, 512)):
    """
    Convert CVAT Pascal VOC XML annotations to MOT Challenge format.
    
    Args:
        xml_dir (str): Directory containing .xml annotation files
        image_dir (str): Directory containing corresponding images
        output_dir (str): Output directory for .txt files
        img_size (tuple): Input size expected by model (w, h)
    """
    # Map filename to track_id based on player jersey or position
    # You can extend this logic using metadata or naming convention
    player_id_map = {
        'player_red_left': 1,
        'player_blue_left': 2,
        'player_red_right': 3,
        'player_blue_right': 4,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    clips = [d for d in os.listdir(xml_dir) if os.path.isdir(os.path.join(xml_dir, d))]
    
    for clip_name in clips:
        xml_clip_path = os.path.join(xml_dir, clip_name)
        img_clip_path = os.path.join(image_dir, clip_name)
        mot_output_file = os.path.join(output_dir, f"{clip_name}.txt")
        
        with open(mot_output_file, 'w') as mot_file:
            frame_idx = 0
            for xml_file in sorted(os.listdir(xml_clip_path)):
                if not xml_file.endswith('.xml'):
                    continue
                
                tree = ET.parse(os.path.join(xml_clip_path, xml_file))
                root = tree.getroot()
                
                # Get image dimensions
                img_path = os.path.join(img_clip_path, xml_file.replace('.xml', '.jpg'))
                if os.path.exists(img_path):
                    img_h, img_w = cv2.imread(img_path).shape[:2]
                else:
                    img_w, img_h = img_size[0], img_size[1]  # Fallback
                
                for obj in root.findall('object'):
                    name = obj.find('name').text.lower().strip()
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    # Assign track_id based on naming convention or manual mapping
                    # Example: "player_1", "player_2", etc.
                    try:
                        track_id = int(name.split('_')[-1])  # e.g., "player_1" -> 1
                    except:
                        print(f"Warning: Could not extract track_id from {name}")
                        continue

                    class_id = 0  # Only 'player' class

                    # Normalize coordinates
                    x_center = ((xmin + xmax) / 2) / img_w
                    y_center = ((ymin + ymax) / 2) / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h

                    # Write to MOT format
                    mot_file.write(
                        f"{frame_idx} {track_id} "
                        f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
                        f"1 -1 -1 -1\n"
                    )
                
                frame_idx += 1
        
        print(f"✅ Converted {clip_name} to MOT format")

# Usage
if __name__ == "__main__":
    xml_dir = "annotations/xml/"      # CVAT export per clip
    image_dir = "images/train/"       # Original frames
    output_dir = "labels_with_ids/train/"
    
    xml_to_mot(xml_dir, image_dir, output_dir)
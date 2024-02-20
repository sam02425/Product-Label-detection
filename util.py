import torch
from transformers import BertTokenizer, BertForMaskedLM
import string
import easyocr

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def read_product_label(product_label_crop):
    # Perform BERT-based text recognition on the product label crop
    inputs = tokenizer(product_label_crop, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    # Extract the predicted labels and convert them to human-readable text
    predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().numpy()
    predicted_text = tokenizer.decode(predicted_labels)

    # Return the recognized text and confidence score (you may need to fine-tune this process)
    return predicted_text, 0.9  # Replace 0.9 with an appropriate confidence score

def product_label_complies_format(text):
    # Implement checks for compliance with the expected format
    # Return True if the format is compliant, False otherwise
    return len(text) > 0  # Replace with your compliance checks

def get_product(product_label, product_track_ids):
    # Implement logic to retrieve product information based on coordinates
    # Return the product information (ID, name, etc.)
    x1, y1, x2, y2, score, class_id = product_label
    product_center_x = (x1 + x2) / 2
    product_center_y = (y1 + y2) / 2

    for track_id, track in enumerate(product_track_ids):
        x_track1, y_track1, x_track2, y_track2, _ = track
        if x_track1 <= product_center_x <= x_track2 and y_track1 <= product_center_y <= y_track2:
            return {"product_id": track_id, "product_name": "Example Product"}  # Modify with actual product information
        
    return {"product_id": -1, "product_name": "Unknown Product"}

def crop_product_label(frame, product_label_coordinates):
    """
    Crop the product label from the frame using the provided coordinates.

    Args:
        frame (numpy.ndarray): The input frame.
        product_label_coordinates (tuple): A tuple containing (x1, y1, x2, y2) coordinates.

    Returns:
        numpy.ndarray: The cropped product label region.
    """
    x1, y1, x2, y2 = product_label_coordinates
    product_label_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    return product_label_crop



def get_product_label(product_label_crop):
    # Implement logic to retrieve additional product label information
    # For example, you can perform custom processing or use external models
    # Return the product label information (ID, name, etc.)
    return {"product_label_id": 1, "product_label_name": "Example Product Label"}

# Assuming you have a loop over frames and products
# Assuming you have a loop over frames and products
results = {}

# Loop over frames
for frame_nmr in range(num_frames):
    results[frame_nmr] = {}

    # Assuming you have a loop over product labels in the frame
    for product_label_id in range(num_product_labels_in_frame):
        # Crop the product label from the frame
        product_label_coordinates = get_product_label_coordinates(frame_nmr, product_label_id)  # Replace with your actual logic
        product_label_crop = crop_product_label(frame, product_label_coordinates)

        # Read product label using BERT-based OCR
        recognized_text, product_label_score = read_product_label(product_label_crop)

        # Check if the recognized text complies with the expected format
        if product_label_complies_format(recognized_text):
            # Retrieve additional product label information
            product_label_info = get_product_label(product_label_crop)

            # Store the results
            results[frame_nmr][product_label_id] = {"recognized_text": recognized_text,
                                                    "product_label_score": product_label_score,
                                                    "product_label_info": product_label_info}


# Process results as needed
            

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'product_id', 'product_bbox',
                                                'product_label_bbox', 'product_label_bbox_score', 'product_label_info',
                                                'confidence_score'))

        for frame_nmr in results.keys():
            for product_id in results[frame_nmr].keys():
                print(results[frame_nmr][product_id])
                if 'product' in results[frame_nmr][product_id].keys() and \
                   'product_label' in results[frame_nmr][product_id].keys() and \
                   'text' in results[frame_nmr][product_id]['product_label'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            product_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][product_id]['product']['bbox'][0],
                                                                results[frame_nmr][product_id]['product']['bbox'][1],
                                                                results[frame_nmr][product_id]['product']['bbox'][2],
                                                                results[frame_nmr][product_id]['product']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][product_id]['product_label']['bbox'][0],
                                                                results[frame_nmr][product_id]['product_label']['bbox'][1],
                                                                results[frame_nmr][product_id]['product_label']['bbox'][2],
                                                                results[frame_nmr][product_id]['product_label']['bbox'][3]),
                                                            results[frame_nmr][product_id]['product_label']['bbox_score'],
                                                            results[frame_nmr][product_id]['product_label']['text'],
                                                            results[frame_nmr][product_id]['product_label']['text_score'])
                            )
        f.close()

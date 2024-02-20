import csv
import numpy as np
from scipy.interpolate import interp1d


def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    product_ids = np.array([int(float(row['product_id'])) for row in data])
    product_bboxes = np.array([list(map(float, row['product_bbox'][1:-1].split())) for row in data])
    product_label_bboxes = np.array([list(map(float, row['product_label_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_product_ids = np.unique(product_ids)
    for product_id in unique_product_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['product_id'])) == int(float(product_id))]
        print(frame_numbers_, product_id)

        # Filter data for a specific product ID
        product_mask = product_ids == product_id
        product_frame_numbers = frame_numbers[product_mask]
        product_bboxes_interpolated = []
        product_label_bboxes_interpolated = []

        first_frame_number = product_frame_numbers[0]
        last_frame_number = product_frame_numbers[-1]

        for i in range(len(product_bboxes[product_mask])):
            frame_number = product_frame_numbers[i]
            product_bbox = product_bboxes[product_mask][i]
            product_label_bbox = product_label_bboxes[product_mask][i]

            if i > 0:
                prev_frame_number = product_frame_numbers[i-1]
                prev_product_bbox = product_bboxes_interpolated[-1]
                prev_product_label_bbox = product_label_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_product_bbox, product_bbox)), axis=0, kind='linear')
                    interpolated_product_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_product_label_bbox, product_label_bbox)), axis=0, kind='linear')
                    interpolated_product_label_bboxes = interp_func(x_new)

                    product_bboxes_interpolated.extend(interpolated_product_bboxes[1:])
                    product_label_bboxes_interpolated.extend(interpolated_product_label_bboxes[1:])

            product_bboxes_interpolated.append(product_bbox)
            product_label_bboxes_interpolated.append(product_label_bbox)

        for i in range(len(product_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['product_id'] = str(product_id)
            row['product_bbox'] = ' '.join(map(str, product_bboxes_interpolated[i]))
            row['product_label_bbox'] = ' '.join(map(str, product_label_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['product_label_bbox_score'] = '0'
                row['product_label'] = '0'
                row['product_label_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['product_id'])) == int(float(product_id))][0]
                row['product_label_bbox_score'] = original_row['product_label_bbox_score'] if 'product_label_bbox_score' in original_row else '0'
                row['product_label'] = original_row['product_label'] if 'product_label' in original_row else '0'
                row['product_label_score'] = original_row['product_label_score'] if 'product_label_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data


# Load the CSV file
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'product_id', 'product_bbox', 'product_label_bbox', 'product_label_bbox_score', 'product_label', 'product_label_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

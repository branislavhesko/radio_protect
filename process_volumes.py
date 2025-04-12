import glob
import os
import numpy as np
import pydicom
import cv2
import h5py


ROI_MAPPING = {
    "CTV_Low": 1,
    "CTV_High": 2,
    "PTV_Low": 3,
    "PTV_High": 4,
    "GTV": 5,
    "Lungs": 6,
}



def process_ct_volumes(data_path):
    """Process CT volumes and return organized volume data."""
    dcms = glob.glob(os.path.join(data_path, "*.dcm"))
    cts = list(filter(lambda x: "CT." in x, dcms))
    
    study_images = {}
    for c in cts:
        dcm = pydicom.dcmread(c)
        if np.array(dcm.pixel_array).shape == (512, 512):
            if dcm.SeriesInstanceUID not in study_images:
                study_images[dcm.SeriesInstanceUID] = []
            study_images[dcm.SeriesInstanceUID].append({
                'pixel_array': np.array(dcm.pixel_array),
                'z_position': dcm.ImagePositionPatient[2],
                'slice_uid': dcm.SOPInstanceUID,
                'image_position': dcm.ImagePositionPatient,
                'pixel_spacing': dcm.PixelSpacing
            })
    
    processed_volumes = {}
    for study_uid, slices in study_images.items():
        # Sort slices by z-position
        sorted_slices = sorted(slices, key=lambda x: x['z_position'])
        volume = np.stack([cv2.resize(x['pixel_array'], (512, 512)) for x in sorted_slices])
        slice_uids = [x['slice_uid'] for x in sorted_slices]
        processed_volumes[study_uid] = {
            'volume': volume,
            'slice_uids': slice_uids,
            'z_positions': [x['z_position'] for x in sorted_slices],
            'image_positions': [x['image_position'] for x in sorted_slices],
            'pixel_spacings': [x['pixel_spacing'] for x in sorted_slices]
        }
    
    return processed_volumes

def patient_to_pixel_coords(points, image_position, pixel_spacing):
    """Convert points from patient coordinates to pixel coordinates."""
    # Convert to pixel coordinates
    j = (points[:, 0] - image_position[0]) / pixel_spacing[1]
    i = (points[:, 1] - image_position[1]) / pixel_spacing[0]  
    
    # Round to nearest integer and clip to image bounds
    i = np.clip(np.round(i), 0, 511).astype(int)
    j = np.clip(np.round(j), 0, 511).astype(int)
    
    return np.column_stack((j, i))


def process_contours(data_path):
    """Process contour data and return organized contour information."""
    contour_data = {}
    for file in glob.glob(os.path.join(data_path, "RS*.dcm")):
        ds = pydicom.dcmread(file)
        if hasattr(ds, 'ROIContourSequence'):
            for roi_contour in ds.ROIContourSequence:
                roi_number = roi_contour.ReferencedROINumber
                roi_name = None
                for roi in ds.StructureSetROISequence:
                    if roi.ROINumber == roi_number:
                        roi_name = roi.ROIName
                        break
                if roi_name not in ROI_MAPPING:
                    continue
                if hasattr(roi_contour, 'ContourSequence'):
                    for contour in roi_contour.ContourSequence:
                        if not contour.ContourImageSequence[0].ReferencedSOPInstanceUID in contour_data:
                            contour_data[contour.ContourImageSequence[0].ReferencedSOPInstanceUID] = []
                        contour_data[contour.ContourImageSequence[0].ReferencedSOPInstanceUID].append({
                            'points': np.array(contour.ContourData).reshape(-1, 3),
                            'roi_name': roi_name,
                            'roi_number': ROI_MAPPING[roi_name]
                        })
    return contour_data

def main():
    # Configuration
    PATIENT = "1"  # You can change this as needed
    DATA_PATH = f"./data/SAMPLE_00{PATIENT}/"
    
    # Process CT volumes
    print("Processing CT volumes...")
    volumes = process_ct_volumes(DATA_PATH)
    
    # Process contours
    print("Processing contours...")
    contours = process_contours(DATA_PATH)
    
    # Combine data and save
    for study_uid, volume_data in volumes.items():
        # Create masks for each slice
        masks = np.zeros_like(volume_data['volume'], dtype=np.uint8)
        
        # Match contours to slices
        for i, (slice_uid, image_position, pixel_spacing) in enumerate(zip(
            volume_data['slice_uids'],
            volume_data['image_positions'],
            volume_data['pixel_spacings']
        )):
            if slice_uid in contours:
                single_mask_contours = contours[slice_uid]
                mask = np.zeros((512, 512), dtype=np.uint8)

                for contour in single_mask_contours:
                    points = contour['points']                    
                    # Convert points from patient coordinates to pixel coordinates
                    points_2d = patient_to_pixel_coords(points, image_position, pixel_spacing)
                    print(points_2d.shape)                # Create mask using the converted points
                    cv2.fillPoly(mask, [points_2d], contour['roi_number'])
                    masks[i] = mask
        
        combined_array = np.stack([volume_data['volume'], masks], axis=-1)
        
        # Save combined data in NPY format
        output_file_npy = f"volume_contours_{PATIENT}_{study_uid}.npy"
        np.save(output_file_npy, combined_array)
        
        print(f"Saved combined data to {output_file_npy}")

if __name__ == "__main__":
    main() 
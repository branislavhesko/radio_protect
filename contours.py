import glob
import os
import pathlib

import cv2
import pydicom
from matplotlib import pyplot as plt
import numpy as np

import vedo


DATA_PATH = "./data/SAMPLE_002/"


for file in glob.glob(DATA_PATH + "RS*.dcm"):
    ds = pydicom.dcmread(file)
    if hasattr(ds, 'SOPInstanceUID'):
        print(f"Found structure set file: {file}")
        
        # For visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.tab20(np.linspace(0, 1, 20))  
        
        # Extract ROI contour data
        if hasattr(ds, 'ROIContourSequence'):
            for i, roi_contour in enumerate(ds.ROIContourSequence):
                # Get ROI number to cross-reference with ROI name
                roi_number = roi_contour.ReferencedROINumber
                
                # Find the ROI name from the structure set ROI sequence
                roi_name = None
                for roi in ds.StructureSetROISequence:
                    if roi.ROINumber == roi_number:
                        roi_name = roi.ROIName
                        break
                
                print(f"ROI #{roi_number}: {roi_name}")
                
                # Extract contour data if available
                if hasattr(roi_contour, 'ContourSequence'):
                    contour_count = len(roi_contour.ContourSequence)
                    print(f"  - Contains {contour_count} contours")
                    
                    # Collect all points for this ROI for visualization
                    all_points = []
                    
                    # Process all contours for this ROI
                    for j, contour in enumerate(roi_contour.ContourSequence):
                        contour_data = contour.ContourData
                        points = np.array(contour_data).reshape(-1, 3)
                        all_points.append(points)
                        
                        # Example: extract points from first contour
                        if j == 0 and len(points) > 0:
                            print(f"  - First contour has {len(points)} points")
                    
                    # Visualize this ROI's contours
                    color = colors[i % len(colors)]
                    for points in all_points:
                        if len(points) > 0:
                            ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                                    color=color, alpha=0.7, linewidth=1)
                            
            # Set plot properties
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('3D Visualization of ROI Contours')
            plt.tight_layout()
            plt.show()
            
            # Alternative visualization with vedo
            print("\nGenerating 3D visualization with vedo...")
            meshes = []
            for i, roi_contour in enumerate(ds.ROIContourSequence):
                # Find ROI name
                roi_number = roi_contour.ReferencedROINumber
                roi_name = None
                for roi in ds.StructureSetROISequence:
                    if roi.ROINumber == roi_number:
                        roi_name = roi.ROIName
                        break
                
                if hasattr(roi_contour, 'ContourSequence') and len(roi_contour.ContourSequence) > 0:
                    # Collect all contour points for this ROI
                    all_contour_points = []
                    for contour in roi_contour.ContourSequence:
                        points = np.array(contour.ContourData).reshape(-1, 3)
                        if len(points) > 2:  # Need at least 3 points to form a contour
                            all_contour_points.append(points)
                    
                    if all_contour_points:
                        color = colors[i % len(colors)][:3] 
                        for points in all_contour_points:
                            # Create a vedo polygon from points
                            poly = vedo.Line(points, closed=True)
                            poly.color(color)
                            meshes.append(poly)
            
            if meshes:
                # Create a vedo plotter and show all meshes
                plt = vedo.Plotter()
                plt.show(meshes, viewup="z", axes=1)
        else:
            print("No ROIContourSequence found in the structure set")
        break
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

# Open the extracted DEM file
with rasterio.open("elevation.tif") as src:
    elevation = src.read(1)
    # Mask out no-data values if defined
    elevation = np.ma.masked_equal(elevation, src.nodata)
    transform = src.transform
    bounds = src.bounds
    print("DEM Bounds:", bounds)
    
    # Define the source CRS (lunar geographic) and target CRS (DEM's polar stereographic)
    source_crs = "+proj=longlat +R=1737400 +no_defs"
    target_crs = src.crs.to_string()
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    # First marker: (lon, lat) = (222, -89)
    marker1_lon = 218.0
    marker1_lat = -89.480
    marker1_x, marker1_y = transformer.transform(marker1_lon, marker1_lat)
    marker1_row, marker1_col = src.index(marker1_x, marker1_y)
    print(f"Marker 1 (222, -89) projects to (x={marker1_x:.2f}, y={marker1_y:.2f}) and pixel (col={marker1_col}, row={marker1_row})")
    
    #Kerr
    marker2_lon = 199.49000
    marker2_lat = -89.58060
    marker2_x, marker2_y = transformer.transform(marker2_lon, marker2_lat)
    marker2_row, marker2_col = src.index(marker2_x, marker2_y)
    print(f"Marker 2 (199.49, -89.58060) projects to (x={marker2_x:.2f}, y={marker2_y:.2f}) and pixel (col={marker2_col}, row={marker2_row})")
    
# Create the heat map plot
plt.figure(figsize=(10, 8))
heatmap = plt.imshow(elevation, cmap='viridis', origin='upper')
plt.colorbar(heatmap, label='Elevation (m)')
plt.title("Heat Map of Extracted LDEM Area with Markers")
plt.xlabel("Column")
plt.ylabel("Row")

# Plot the first marker as a red 'X'
plt.scatter(marker1_col, marker1_row, s=100, c='red', marker='x', label="Marker (222, -89)")

# Plot the second marker as a blue 'o'
plt.scatter(marker2_col, marker2_row, s=100, c='blue', marker='o', label="Marker (222, -89.58060)")

# Annotate the second marker with its elevation value
plt.text(marker2_col + 10, marker2_row, "1128.94 m", color="blue", fontsize=12, weight="bold")

plt.legend()
plt.tight_layout()

# Save and display the output
plt.savefig("extracted_area_heatmap_with_markers.png", dpi=300)
plt.show()








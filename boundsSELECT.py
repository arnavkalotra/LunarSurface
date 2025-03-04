import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the base map image
# (Assuming you've exported the PNG from map_georef.tif as "map_georef.png")
base_img = Image.open("quickmap-lroc.png").convert("RGBA")
base_arr = np.array(base_img)

# Open the aligned DEM and read the elevation data
with rasterio.open("LOLA_dem_aligned.tif") as src:
    elevation = src.read(1)
    # Mask out no-data values if present
    elevation = np.ma.masked_equal(elevation, src.nodata)

# Normalize elevation for colormap application
elev_min = elevation.min()
elev_max = elevation.max()
norm_elev = (elevation - elev_min) / (elev_max - elev_min)

# Apply a colormap (e.g., viridis) to create an RGBA image
cmap = plt.get_cmap("viridis")
heatmap_rgba = cmap(norm_elev.filled(0))  # shape: (rows, cols, 4), values in [0,1]
# Convert to 8-bit integer values
heatmap_img = (heatmap_rgba * 255).astype(np.uint8)

# Convert heatmap array to a PIL image
heatmap_pil = Image.fromarray(heatmap_img, mode="RGBA")

# Set the heatmap opacity (alpha) lower, e.g., 50%
# We'll adjust the alpha channel of the heatmap
alpha = 128  # out of 255 (50% opacity)
heatmap_data = np.array(heatmap_pil)
heatmap_data[..., 3] = alpha  # set all alpha values to 128
heatmap_pil = Image.fromarray(heatmap_data, mode="RGBA")

# Composite the two images: base map and heatmap overlay
combined = Image.alpha_composite(base_img, heatmap_pil)

# Save and display the result
combined.save("combined_heatmap.png")
combined.show()

# Optionally, display with matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(combined)
plt.axis("off")
plt.title("Base Map with DEM Heatmap Overlay")
plt.tight_layout()
plt.savefig("combined_heatmap_matplotlib.png", dpi=300)
plt.show()

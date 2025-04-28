import rasterio
from rasterio.crs import CRS
from pyproj import Transformer
from PIL import Image, ImageDraw

def plot_coordinate_on_geotiff(
    geotiff_path,
    out_png_path,
    lon,
    lat,
    marker_size=5
):
    """
    Opens a georeferenced GeoTIFF (1163x820) that uses the following VRT-based georeference:
      GeoTransform: -29261.572648417437, 30.00045813234842, 0, -6680.256860031139, 0, -30.00045813234842
      SRS: +proj=stere +a=1737400 +b=1737400 +lat_0=-90.0 +lon_0=0.0 +x_0=0 +y_0=0 +k=1 +units=m +no_defs
    It transforms input coordinates from Moon lat/lon (using +proj=longlat +R=1737400 +no_defs)
    into the TIFF's CRS, converts them to pixel coordinates, draws a small white square marker,
    and saves the annotated image as a PNG.
    
    :param geotiff_path: Path to the input GeoTIFF (e.g. 'map_georef.tif')
    :param out_png_path: Path to save the annotated PNG
    :param lon: Longitude of the point (degrees) in Moon lat/lon system
    :param lat: Latitude of the point (degrees) in Moon lat/lon system
    :param marker_size: Half-size (in pixels) of the square marker
    """
    # Open the GeoTIFF to read its metadata
    with rasterio.open(geotiff_path) as ds:
        tiff_crs = ds.crs
        
        # Define the Moon lat/lon CRS
        moon_latlon_crs = CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
        
        # Create a transformer from Moon lat/lon to the TIFF's CRS
        transformer = Transformer.from_crs(moon_latlon_crs, tiff_crs, always_xy=True)
        
        # Transform the input (lon, lat) to (x, y) in the TIFF's CRS (meters)
        x, y = transformer.transform(lon, lat)
        
        # Convert the (x, y) coordinate to pixel coordinates.
        # ds.index() returns (row, col)
        row, col = ds.index(x, y)
        
        print(f"Transformed (lon={lon}, lat={lat}) => (x={x:.2f}, y={y:.2f}) => pixel=(col={col}, row={row})")
        
        # Open the same GeoTIFF with Pillow to draw the marker.
        with Image.open(geotiff_path).convert("RGBA") as img:
            draw = ImageDraw.Draw(img)
            
            # Calculate the marker rectangle coordinates
            left   = col - marker_size
            right  = col + marker_size
            top    = row - marker_size
            bottom = row + marker_size
            
            # Draw a small white square marker
            draw.rectangle([left, top, right, bottom], fill="blue")
            
            # Save the annotated image as PNG
            img.save(out_png_path)
            print(f"Saved annotated PNG to {out_png_path}")

if __name__ == "__main__":
    # Example usage: adjust lon and lat as needed.
    # Here, lon and lat are in Moon lat/lon (using +proj=longlat +R=1737400 +no_defs)
    plot_coordinate_on_geotiff(
        geotiff_path="LOLA_dem_aligned.tif",  # This GeoTIFF should be created from the above VRT.
        out_png_path="annotated_map.png",
        lon=222,  
        lat=-89,   
        marker_size=6
    )

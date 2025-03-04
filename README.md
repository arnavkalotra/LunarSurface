CREATE SIMULATION OF LUNAR SURFACE USING MODEL DISTILATION 

Long term: use image distillation to generate 3D ai environment to optimize pathfinding for lunar habitation in a 2040 mission.
This is in conjuction with NASA RASC-AL competition as it will originally serve as a ice rover path optimization algorithm.
This environment can then be set up for a variety of scenarios

 

DATASET  

LDEM_83S_10MPP_ADJ.TIF - 4.8G    [LOLA ELEVATION MAP 10 m/pixel] 

LDSM_83S_10MPP_ADJ.TIF - 7.1G.  [LOLA SLOP MAP 10 m/pixel] 

 

LOLA 

“In a 50km polar orbit, pulsing the laser at 28 Hz creates an ~50m-wide swatch of five topographic profiles. Swaths will have 1.25km separation at the equator, with [complete polar coverage beyond +/-86 degrees latitude.]” 

south polar stereographic X/Y coordinates in meters and in the MOON_ME reference frame of the JPL DE421 ephemeris. 

 

AI MODEL DISTILLATION 

Use error data for slope and elevation 

Use hillshade data as well to accurately map surface slope difference  

Use effective resolution data for ? 

 


 

Utilize LOLA dataset as metadata for visual map 

Establish habitat location and ice (based on previous imaging) 

Allign properties to coordinates  

 
 



 

Simulation  

High res scaled image by using distillation , fitted with meta data from LOLA elevations 

Topical 3d view with contours pertaining to elevation and slope data  

Use LCROSS spectrometer data to generate wavelength data in icey regions 

Rover will take time to mine and store ice 

Multiple runs will be recorded 

Set multiple realistic parameters (i.e solar recharching, maintenance ) 

Machine learning model will learn as more runs are recorded  

ENVIRONMENT 

Left: -29261.572648417437  

Right: 5628.960159503775  

Top: -6680.256860031139  

Bottom: -31280.632528556842  

Converted to Moon long/lat:  

 

Top Left (lon, lat): (-102.85992303721196, -89.01021245642394)  

Top Right (lon, lat): (139.88162995401004, -89.71191838603046)  

Bottom Left (lon, lat): (-136.91008487880595, -88.58750983238276)  

Bottom Right (lon, lat): (169.7987828469446, -88.95189095690426) 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

File upload nodes 

 

Use python code to accurate remove background for precise latitude, longitude alignment ( USED REMBG FOR PRECISE CROPPING) 

 

Calibrate LOLA MAP using gdal ( vrt files combined with png) (pixel scale of 10m/px) 

 

 Align elevation data with slope data 

 

 

 

 

WORKING ON 

Calibrating LROC png with vrt in order to have precise coordinate system ( using gdal) [COMPLETE!] 

Sift through LOLA sets and line up slope, roughness. 

 

 

 

 

 

 

 

 

 

 

 

 

https://www.nasa.gov/general/what-is-lcross-the-lunar-crater-observation-and-sensing-satellite/ 

 

https://planetarydata.jpl.nasa.gov/img/data/lcross/LCRO_0001/DATA/20091009113022_IMPACT/MIR1/CAL/ 

 

LOLA DATA 

https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m 

ELEVATION 

https://science.nasa.gov/mission/lro/lola/ 

LARGE DATASET ( includes slope and roughness) 

https://pgda.gsfc.nasa.gov/products/90 

 

 

Temperature and Slope, as well as MAP CSV 

https://quickmap.lroc.asu.edu/?prjExtent=-3004075.862069%2C-1737400%2C3004075.862069%2C1737400&selectedFeature=3489%2C8&queryOpts=N4IgLghgRiBcIBMKRAXyA&shadowsType=all&layers=NrBsFYBoAZIRnpEBmZcAsjYIHYFcAbAyAbwF8BdC0yioA&proj=10 

 

 

 

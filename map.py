import os
import requests
import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import folium
from folium.plugins import HeatMap

# Centar mape
CENTER = [44.787197, 20.457273]

# WorldPop raster (Srbija, gustina populacije 2020, 1 km rezolucija)
RASTER_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/SRB/srb_ppp_2020_1km_Aggregated.tif"
RASTER_PATH = "srb_ppp_2020_1km_Aggregated.tif"

# Preuzimanje rastera ako ne postoji u folderu
if not os.path.exists(RASTER_PATH):
    with requests.get(RASTER_URL, stream=True) as r:
        r.raise_for_status()
        with open(RASTER_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# Poligon koji obuhvata područje Beograda
poly = Polygon([
    (20.18, 44.58),
    (20.18, 44.98),
    (20.75, 44.98),
    (20.75, 44.58)
])
gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly])

# Maskiranje rastera – izdvajamo samo Beograd
with rasterio.open(RASTER_PATH) as src:
    out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True)
    pop = out_image[0]

# Priprema tačaka: lat, lon, gustina
rows, cols = np.where(pop > 0)
points = []
for r, c in zip(rows, cols):
    lon, lat = rasterio.transform.xy(out_transform, r, c)
    val = float(pop[r, c])
    points.append([lat, lon, val])

# Normalizacija gustine (0 – 1)
max_val = max(p[2] for p in points)
heat_points = [[p[0], p[1], p[2] / max_val] for p in points]

# Kreiranje mape
m = folium.Map(location=CENTER, zoom_start=11, tiles="cartodbpositron")

# Heatmap sloj
HeatMap(
    heat_points,
    radius=28,
    blur=38,
    min_opacity=0.35,
    max_opacity=0.95
).add_to(m)

# Dodavanje tačaka proporcionalne veličine
for lat, lon, val in heat_points:
    size = 2 + val * 18  # 2–20 px
    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        stroke=False
    ).add_to(m)

# Snimanje mape
m.save("beograd_gustina_populacije_tacke.html")
print("✔ Gotovo! Otvori: beograd_gustina_populacije_tacke.html")

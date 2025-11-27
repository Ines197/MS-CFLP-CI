import os
import requests
import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import folium
from folium.plugins import HeatMap
import osmnx as ox

# Centar mape
CENTER = [44.787197, 20.457273]

# WorldPop raster za Srbiju — 2020, 1 km rezolucija
RASTER_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/SRB/srb_ppp_2020_1km_Aggregated.tif"
RASTER_PATH = "srb_ppp_2020_1km_Aggregated.tif"

# Preuzimanje rastera ako još nije skinut
if not os.path.exists(RASTER_PATH):
    with requests.get(RASTER_URL, stream=True) as r:
        r.raise_for_status()
        with open(RASTER_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

#Poligon (bounding box) koji pokriva Beograd
poly = Polygon([
    (20.18, 44.58),
    (20.18, 44.98),
    (20.75, 44.98),
    (20.75, 44.58)
])
gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly])

# Maskiranje rastera na Beograd
with rasterio.open(RASTER_PATH) as src:
    out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True)
    pop = out_image[0]

# Menjamo tacke
rows, cols = np.where(pop > 0)
points = []
for r, c in zip(rows, cols):
    lon, lat = rasterio.transform.xy(out_transform, r, c)
    val = float(pop[r, c])
    points.append([lat, lon, val])

# Normalizacija gustine (0–1)
max_val = max(p[2] for p in points)
heat_points = [[p[0], p[1], p[2] / max_val] for p in points]

# Kreiraj mapu
m = folium.Map(location=CENTER, zoom_start=11, tiles="cartodbpositron")

# Heatmap populacije
HeatMap(
    heat_points,
    radius=28,
    blur=38,
    min_opacity=0.35,
    max_opacity=0.95
).add_to(m)

# Učitavanje pumpi iz OpenStreetMap
pumpe = ox.features_from_place(
    "Belgrade, Serbia",
    tags={"amenity": "fuel"}
)

pump_points = []
for _, row in pumpe.iterrows():
    if row.geometry.geom_type == "Point":
        pump_points.append([row.geometry.y, row.geometry.x])


# Funkcija: gustina populacije najbliže raster ćelije pumpi
def nearest_pop_weight(lat, lon):
    d = [abs(lat - p[0]) + abs(lon - p[1]) for p in heat_points]
    idx = np.argmin(d)
    return heat_points[idx][2]   # vrednost 0–1

# Dodavanje pumpi — veličina proporcionalna gustini populacije
for lat, lon in pump_points:
    w = nearest_pop_weight(lat, lon)
    size = 4 + w * 14  # 4–18 px
    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        color="darkred",
        fill=True,
        fill_color="darkred",
        fill_opacity=0.85,
        stroke=False
    ).add_to(m)

# Snimanje mape
m.save("beograd_gustina_populacije_pumpe.html")
print("✔ Gotovo! Otvori mapu: beograd_gustina_populacije_pumpe.html")
print("jeeej")
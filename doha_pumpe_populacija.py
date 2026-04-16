import os
import requests
import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import folium
import osmnx as ox
import pandas as pd


# ================================================================
# SETTINGS
# ================================================================

CENTER = [25.2854, 51.5310]   # centar Dohe

RASTER_URL = "https://data.humdata.org/dataset/59b0d318-966d-4688-a547-c5126903eea2/resource/21506e5c-15c1-4185-bae4-ed4cd34f79dd/download/qat_pop_2020_cn_1km.tif"
RASTER_PATH = "qat_pop_2020_cn_1km.tif"

POP_THRESHOLD = 50
NUM_INSTANCES = 20


# ================================================================
# HAVERSINE
# ================================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def offset_point(lat, lon, meters=30, rng=None):
    angle = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(0, meters)
    d_lat = (dist * np.cos(angle)) / 111320
    d_lon = (dist * np.sin(angle)) / (111320 * np.cos(np.radians(lat)))
    return lat + d_lat, lon + d_lon


# ================================================================
# 1) PREUZIMANJE RASTERA
# ================================================================
if not os.path.exists(RASTER_PATH):
    print("Preuzimam raster...")
    with requests.get(RASTER_URL, stream=True) as r:
        r.raise_for_status()
        with open(RASTER_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)


# ================================================================
# 2) UČITAVANJE POPULACIJE ZA DOHU
# ================================================================
poly = Polygon([
    (51.35, 25.15),
    (51.35, 25.42),
    (51.67, 25.42),
    (51.67, 25.15)
])

gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly])

with rasterio.open(RASTER_PATH) as src:
    out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True)
    pop = out_image[0]

rows, cols = np.where(pop > 0)
base_points = []

for r, c in zip(rows, cols):
    lon, lat = rasterio.transform.xy(out_transform, r, c)
    val = float(pop[r, c])
    base_points.append([lat, lon, val])

print("Ukupno raster tačaka:", len(base_points))


# ================================================================
# 3) UČITAVANJE PUMPI
# ================================================================
print("Učitavam pumpe iz OSM...")
pumpe = ox.features_from_place("Doha, Qatar", tags={"amenity": "fuel"})

pump_points = []
for _, row in pumpe.iterrows():
    geom = row.geometry

    if geom.geom_type == "Point":
        pump_points.append([geom.y, geom.x])

    elif geom.geom_type in ["Polygon", "MultiPolygon"]:
        centroid = geom.centroid
        pump_points.append([centroid.y, centroid.x])

print("Broj pumpi:", len(pump_points))


# ================================================================
# 4) GENERISANJE JEDNE INSTANCE
# ================================================================
def generate_instance(instance_id):
    rng = np.random.default_rng(seed=instance_id)

    print(f"\n--- Generišem instancu doha_{instance_id} ---")

    # ------------------------------------------------------------
    # FILTER
    # ------------------------------------------------------------
    filtered_points = [p[:] for p in base_points if p[2] >= POP_THRESHOLD]
    print("Nakon filtera:", len(filtered_points))

    # ------------------------------------------------------------
    # DELJENJE 30% NAJVEĆIH TAČAKA
    # ------------------------------------------------------------
    sorted_points = sorted(filtered_points, key=lambda x: x[2], reverse=True)

    k = int(0.30 * len(sorted_points))
    top_points = sorted_points[:k]
    rest_points = sorted_points[k:]

    new_points = []
    for lat, lon, val in top_points:
        for _ in range(3):
            lat2, lon2 = offset_point(lat, lon, meters=30, rng=rng)
            new_points.append([lat2, lon2, val / 3])

    filtered_points = rest_points + new_points
    print("Posle deljenja:", len(filtered_points))

    # ------------------------------------------------------------
    # GOODS
    # ------------------------------------------------------------
    Goods = [int(round(p[2])) for p in filtered_points]
    total_goods = sum(Goods)
    print("Ukupna populacija:", total_goods)

    # ------------------------------------------------------------
    # CAPACITY = 5 * UKUPNA POPULACIJA
    # ------------------------------------------------------------
    center_lat, center_lon = CENTER
    distances_to_center = [
        haversine(lat, lon, center_lat, center_lon)
        for lat, lon in pump_points
    ]

    max_dist = max(distances_to_center) if distances_to_center else 1
    target_total_capacity = 5 * total_goods

    weights = []
    for d in distances_to_center:
        centrality = 1 - (d / max_dist)
        w = 0.5 + centrality
        weights.append(w)

    weights = np.array(weights, dtype=float)

    if len(weights) > 0:
        weights = weights / weights.sum()
        Capacity = np.floor(weights * target_total_capacity).astype(int)

        diff = target_total_capacity - Capacity.sum()
        for i in range(diff):
            Capacity[i % len(Capacity)] += 1

        Capacity = Capacity.tolist()
    else:
        Capacity = []

    print("Ukupan kapacitet:", sum(Capacity) if Capacity else 0)

    # ------------------------------------------------------------
    # FIXED COST U [5, 20] PO CENTRALNOSTI
    # ------------------------------------------------------------
    FixedCost = []
    if len(pump_points) > 0:
        for d in distances_to_center:
            centrality = 1 - (d / max_dist)
            fc = 5 + centrality * (20 - 5)
            FixedCost.append(int(round(fc)))

    # ------------------------------------------------------------
    # KATEGORIJE POPULACIJE: 1, 2, 3, 4
    # ------------------------------------------------------------
    num_points = len(filtered_points)

    category_values = [1, 2, 3, 4]
    category_probs = [0.45, 0.35, 0.15, 0.05]

    population_category = rng.choice(
        category_values,
        size=num_points,
        p=category_probs
    )

    category_to_color = {
        1: "red",
        2: "blue",
        3: "yellow",
        4: "green"
    }

    category_to_charger = {
        1: "AC nivo 2",
        2: "DC",
        3: "Ultra brzi",
        4: "AC nivo 2"
    }

    colors = np.array([category_to_color[c] for c in population_category])
    charger_type = np.array([category_to_charger[c] for c in population_category])

    # ------------------------------------------------------------
    # DISTANCE MATRIX
    # ------------------------------------------------------------
    pop_coords = np.array([[p[0], p[1]] for p in filtered_points])
    pump_coords = np.array([[p[0], p[1]] for p in pump_points])

    n_pop = len(pop_coords)
    n_pump = len(pump_coords)

    dist_matrix = np.zeros((n_pop, n_pump), dtype=int)

    for i in range(n_pop):
        lat1, lon1 = pop_coords[i]
        lat2 = pump_coords[:, 0]
        lon2 = pump_coords[:, 1]
        dist_matrix[i, :] = np.rint(haversine(lat1, lon1, lat2, lon2)).astype(int)

    # ------------------------------------------------------------
    # INKOMPATIBILNOSTI
    # ------------------------------------------------------------
    incomp_pairs = []

    for i in range(num_points):
        for j in range(i + 1, num_points):
            c1 = population_category[i]
            c2 = population_category[j]

            if (c1 == 1 and c2 in (2, 3, 4)) or (c2 == 1 and c1 in (2, 3, 4)):
                incomp_pairs.append((i + 1, j + 1))
            elif (c1 == 2 and c2 in (3, 4)) or (c2 == 2 and c1 in (3, 4)):
                incomp_pairs.append((i + 1, j + 1))
            elif (c1 == 3 and c2 == 4) or (c1 == 4 and c2 == 3):
                incomp_pairs.append((i + 1, j + 1))

    print("Incompatibilities:", len(incomp_pairs))

    # ------------------------------------------------------------
    # TABELA KATEGORIJA
    # ------------------------------------------------------------
    category_table = pd.DataFrame({
        "Kategorija": [1, 2, 3, 4],
        "Boja na mapi": ["red", "blue", "yellow", "green"],
        "Tip punjača": ["AC nivo 2", "DC", "Ultra brzi", "AC nivo 2"],
        "Verovatnoća": [0.45, 0.35, 0.15, 0.05]
    })

    category_table.to_csv(
        f"doha_{instance_id}_kategorije_populacije.csv",
        index=False,
        encoding="utf-8-sig"
    )

    points_table = pd.DataFrame({
        "lat": [p[0] for p in filtered_points],
        "lon": [p[1] for p in filtered_points],
        "populacija": [p[2] for p in filtered_points],
        "kategorija": population_category,
        "boja": colors,
        "tip_punjaca": charger_type
    })

    points_table.to_csv(
        f"doha_{instance_id}_populacija_po_kategorijama.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # EXPORT U DZN
    # ------------------------------------------------------------
    with open(f"doha_{instance_id}.dzn", "w", encoding="utf-8") as f:
        f.write(f"Warehouses = {n_pump};\n")
        f.write(f"Stores = {n_pop};\n\n")

        f.write("Capacity = [" + ", ".join(map(str, Capacity)) + "];\n")
        f.write("FixedCost = [" + ", ".join(map(str, FixedCost)) + "];\n")
        f.write("Goods = [" + ", ".join(map(str, Goods)) + "];\n")
        f.write("Category = [" + ", ".join(map(str, population_category)) + "];\n")

        f.write("SupplyCost = [|\n")
        for i in range(n_pop):
            f.write(" " + ", ".join(map(str, dist_matrix[i])) + "\n")
        f.write("|];\n\n")

        f.write(f"Incompatibilities = {len(incomp_pairs)};\n")
        f.write("IncompatiblePairs = [| ")
        for a, b in incomp_pairs:
            f.write(f"{a}, {b} | ")
        f.write("];\n")

    # ------------------------------------------------------------
    # HTML MAPA
    # ------------------------------------------------------------
    m = folium.Map(location=CENTER, zoom_start=11, tiles="cartodbpositron")

    pop_layer = folium.FeatureGroup(name="Populacija")
    pump_layer = folium.FeatureGroup(name="Pumpe")

    if filtered_points:
        max_val = max(p[2] for p in filtered_points)

        for i, (lat, lon, val) in enumerate(filtered_points):
            radius = 2 + (val / max_val) * 12

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=colors[i],
                fill=True,
                fill_color=colors[i],
                fill_opacity=0.5,
                stroke=False,
                popup=(
                    f"Kategorija: {population_category[i]} | "
                    f"Tip: {charger_type[i]} | "
                    f"Populacija: {round(val, 2)}"
                )
            ).add_to(pop_layer)

    for lat, lon in pump_points:
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=0.9,
            stroke=False
        ).add_to(pump_layer)

    pop_layer.add_to(m)
    pump_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(f"doha_{instance_id}.html")

    print(f"✔ Sačuvano: doha_{instance_id}.dzn")
    print(f"✔ Sačuvano: doha_{instance_id}.html")


# ================================================================
# 5) GENERISANJE 20 INSTANCI
# ================================================================
for instance_id in range(1, NUM_INSTANCES + 1):
    generate_instance(instance_id)

print("\n✔ Gotovo! Generisano 20 instanci.")
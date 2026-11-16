"""
Pre-computes everything the Globe view needs

  points.json   paleo-coordinates of every fossil occurrence, for every time chunk it overlaps
  earth_<Ma>.jpg  equi-rectangular texture of the continents at each time chunk (wrapped onto the globe)

Plate model: PALEOMAP (C.R. Scotese) rotation file, plate polygons and
coastlines, the same model family used by dinosaurpictures.org/ancient-earth.
The three model files live in the plate-model folder.

Run (only needed when changing dinosaurs.csv or time intervals):

    pip install pygplates pillow numpy
    python reconstruct.py

Alternative without pygplates: 
The GPlates Web Service does the same rotation. 
I didn't set that up though.
Link:
https://gws.gplates.org/reconstruct/reconstruct_points/?lons=..&lats=..&time=..&model=PALEOMAP
"""

import csv
import json
import math # I believe not needed anymore
import os

import numpy as np
import pygplates
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "data", "plate-model") #assuming we will have a data and public folder in the repo
OUT = os.path.join(ROOT, "public", "paleo")

# Time chunks (in million of years ago)
# [T - HALF_BIN, T + HALF_BIN); the globe jumps from chunk to chunk.
FRAMES = list(range(250, 60, -10))  # 250-70, since second arg in range is exclusive
HALF_BIN = 5

TEX_W, TEX_H, SS = 2048, 1024, 2  # file size and supersampling factor


def load_occurrences():
    rows, skipped = [], []
    with open(os.path.join(ROOT, "data", "raw", "dinosaurs.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lng, lat = float(r["lng"]), float(r["lat"])
            if abs(lng) > 180 or abs(lat) > 90:
                skipped.append(r["occurrence_no"])
                continue
            rows.append(
                dict(no=int(r["occurrence_no"]), lng=lng, lat=lat,
                     max_ma=float(r["max_ma"]), min_ma=float(r["min_ma"]))
            )
    return rows, skipped


def assign_plate_ids(rows):
    """Find which present-day plate polygon each fossil site sits on."""
    partitioner = pygplates.PlatePartitioner(
        os.path.join(MODEL, "PALEOMAP_PlatePolygons.gpmlz"),
        pygplates.RotationModel(os.path.join(MODEL, "PALEOMAP_PlateModel.rot")),
    )
    cache = {}
    for r in rows:
        key = (r["lat"], r["lng"])
        if key not in cache:
            plate = partitioner.partition_point(pygplates.PointOnSphere(r["lat"], r["lng"]))
            cache[key] = plate.get_feature().get_reconstruction_plate_id() if plate else 0
        r["pid"] = cache[key]


def reconstruct_points(rows, rotation_model):
    frames = {}
    for t in FRAMES:
        lo, hi = t - HALF_BIN, t + HALF_BIN
        pts = []
        rot_cache = {}
        for r in rows:
            if r["pid"] == 0:
                continue
            if r["max_ma"] < lo or r["min_ma"] >= hi:  # no overlap
                continue
            if r["pid"] not in rot_cache:
                rot_cache[r["pid"]] = rotation_model.get_rotation(float(t), r["pid"])
            lat, lng = (rot_cache[r["pid"]] * pygplates.PointOnSphere(r["lat"], r["lng"])).to_lat_lon()
            pts.append([r["no"], round(lng, 2), round(lat, 2)])
        frames[str(t)] = pts
    return frames


def draw_texture(t, coastlines, rotation_model):
    W, H = TEX_W * SS, TEX_H * SS
    recon = []
    pygplates.reconstruct(coastlines, rotation_model, recon, float(t))
    wrapper = pygplates.DateLineWrapper(0.0)

    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for rg in recon:
        geom = rg.get_reconstructed_geometry()
        if not isinstance(geom, pygplates.PolygonOnSphere):
            continue
        for poly in wrapper.wrap(geom, 2.0):
            xy = [((p.get_longitude() + 180) / 360 * W, (90 - p.get_latitude()) / 180 * H)
                  for p in poly.get_exterior_points()]
            if len(xy) >= 3:
                draw.polygon(xy, fill=255)

    land = np.asarray(mask, dtype=np.float32) / 255
    # halos around the land: shallow shelf seas + a darker coastal rim on land
    shelf = np.asarray(mask.filter(ImageFilter.GaussianBlur(14 * SS)), dtype=np.float32) / 255
    inner = np.asarray(mask.filter(ImageFilter.GaussianBlur(5 * SS)), dtype=np.float32) / 255

    lat = np.linspace(90, -90, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    abslat = np.abs(lat) / 90

    def mix(a, b, w):
        return a[None, None, :] * (1 - w[..., None]) + b[None, None, :] * w[..., None]

    deep = np.array([9, 32, 58], np.float32)
    shallow = np.array([38, 120, 140], np.float32)
    ocean = mix(deep, shallow, np.clip(shelf * 1.6, 0, 1))
    ocean *= (1 - 0.25 * abslat ** 2)[..., None]

    # humid tropics -> dry belts -> cool temperate greens (no polar ice in the Mesozoic apparently. who knew)
    tropic = np.array([86, 138, 78], np.float32)
    dry = np.array([196, 170, 112], np.float32)
    temperate = np.array([104, 134, 84], np.float32)
    w_dry = np.exp(-((abslat - 0.30) / 0.14) ** 2)
    w_temp = np.clip((abslat - 0.42) / 0.25, 0, 1)
    land_col = mix(tropic, dry, w_dry)
    land_col = land_col * (1 - w_temp[..., None]) + temperate[None, None, :] * w_temp[..., None]
    # Interiors a little lighter than coasts
    land_col *= (0.82 + 0.28 * np.clip(inner, 0, 1) ** 3)[..., None]
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32)
    noise = np.asarray(Image.fromarray(noise).resize((W, H), Image.BICUBIC))
    land_col *= (1 + 0.035 * noise)[..., None]

    img = ocean * (1 - land[..., None]) + land_col * land[..., None]
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

    # faint line every 30 degrees
    g = ImageDraw.Draw(img, "RGBA")
    for lon in range(-180, 181, 30):
        x = (lon + 180) / 360 * W
        g.line([(x, 0), (x, H)], fill=(255, 255, 255, 22), width=SS)
    for la in range(-60, 61, 30):
        y = (90 - la) / 180 * H
        g.line([(0, y), (W, y)], fill=(255, 255, 255, 40 if la == 0 else 22), width=SS)

    img = img.resize((TEX_W, TEX_H), Image.LANCZOS)
    img.save(os.path.join(OUT, f"earth_{t}.jpg"), quality=82, optimize=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rotation_model = pygplates.RotationModel(os.path.join(MODEL, "PALEOMAP_PlateModel.rot"))
    rows, skipped = load_occurrences()
    assign_plate_ids(rows)
    no_plate = [r["no"] for r in rows if r["pid"] == 0]
    frames = reconstruct_points(rows, rotation_model)

    with open(os.path.join(OUT, "points.json"), "w") as f:
        json.dump(
            dict(model="PALEOMAP (Scotese)", halfBin=HALF_BIN, frames=FRAMES, points=frames,
                 skipped=dict(invalidCoordinates=skipped, noPlate=no_plate)),
            f, separators=(",", ":"),
        )
    print(f"occurrences: {len(rows)}  invalid coords: {skipped}  not on a plate: {len(no_plate)}")
    for t in FRAMES:
        print(f"  {t} Ma: {len(frames[str(t)])} points")

    coastlines = pygplates.FeatureCollection(os.path.join(MODEL, "PALEOMAP_coastlines.gpmlz"))
    for t in FRAMES:
        draw_texture(t, coastlines, rotation_model)
    print("textures written to", OUT)


if __name__ == "__main__":
    main()

import os 
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
from pathlib import Path
import io
import time
import schedule
import numpy as np
import rasterio
from rasterio.transform import from_origin
import json 
from pymongo import MongoClient


# 1. Main Input Folder
data_path = os.getenv("FIRE_BOT_DATA_DIR", "./data")
INPUT_SHAPES_FOLDER = Path(data_path)

# 2. Output Base Folder
OUTPUT_BASE = INPUT_SHAPES_FOLDER / "monitored fires"


# 3. NASA API Settings
MAP_KEY = "c50d18d9fb014995b1d41f0a0a80929d"

# 4. Sources
SOURCES = [
    "MODIS_NRT",          
    "VIIRS_SNPP_NRT",     
    "VIIRS_NOAA20_NRT",   
    "VIIRS_NOAA21_NRT"    
]

# 5. Settings
PIXEL_SIZE = 0.001
# =================================================

print("--- Dynamic Fire Monitor (with MongoDB Stream) Initialized ---")

# --- Helper 1: Robust Download ---
def robust_get_request(url):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200: return r
            elif r.status_code == 429: 
                print(f"      ! NASA API Busy, waiting 20s...")
                time.sleep(20)
                continue
            time.sleep(1)
        except: time.sleep(2)
    return None

# --- Helper 2: Download Logic ---
def download_fires_for_geometry(gdf, start_date, end_date):
    minx, miny, maxx, maxy = gdf.total_bounds
    area_coords = f"{minx-0.1},{miny-0.1},{maxx+0.1},{maxy+0.1}"
    
    day_range = (end_date - start_date).days + 1
    start_str = start_date.strftime("%Y-%m-%d")
    all_dfs = []

    for source in SOURCES:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{area_coords}/{day_range}/{start_str}"
        r = robust_get_request(url)
        if r and "latitude" in r.text:
            try:
                df = pd.read_csv(io.StringIO(r.text))
                if not df.empty:
                    df['source_api'] = source
                    all_dfs.append(df)
            except: pass
        time.sleep(0.5) 
            
    if not all_dfs: return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

# --- Helper 3: GeoTIFF ---
def save_geotiff(gdf, output_path, pixel_size=PIXEL_SIZE):
    try:
        minx, miny, maxx, maxy = gdf.total_bounds
        width = int((maxx - minx) / pixel_size) + 1
        height = int((maxy - miny) / pixel_size) + 1
        transform = from_origin(minx, maxy, pixel_size, pixel_size)
        
        heatmap, _, _ = np.histogram2d(
            gdf.geometry.y, gdf.geometry.x, 
            bins=(height, width), 
            range=[[miny, maxy], [minx, maxx]]
        )
        heatmap = np.flipud(heatmap) 

        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=heatmap.shape[0], width=heatmap.shape[1],
            count=1, dtype=rasterio.float32, crs='EPSG:4326',
            transform=transform, nodata=0
        ) as dst:
            dst.write(heatmap, 1)
        return True
    except Exception as e:
        print(f"      ! Error creating GeoTIFF: {e}")
        return False

# --- Helper 4: JSON Metadata (Local Storage Backup) ---
def save_fire_json(gdf, area_name, project_name, date_str, output_path):
    try:
        avg_lat = gdf.geometry.y.mean()
        avg_lon = gdf.geometry.x.mean()

        fire_list = []
        for _, row in gdf.iterrows():
            raw_time = str(row.get('acq_time', ''))
            formatted_time = raw_time
            if len(raw_time) == 4 and raw_time.isdigit():
                formatted_time = f"{raw_time[:2]}:{raw_time[2:]}"
            elif len(raw_time) == 3 and raw_time.isdigit():
                formatted_time = f"0{raw_time[:1]}:{raw_time[1:]}"

            fire_obj = {
                "latitude": row.get('latitude'),
                "longitude": row.get('longitude'),
                "brightness": row.get('brightness'),
                "acquisition_time": formatted_time,
                "satellite": row.get('satellite'),
                "instrument": row.get('instrument'),
                "confidence": row.get('confidence'),
                "version": row.get('version'),
                "brightness_band_t31": row.get('bright_t31'),
                "fire_radiative_power": row.get('frp'),
                "source": row.get('source_api')
            }
            
            for k, v in fire_obj.items():
                if isinstance(v, (np.integer, np.floating)):
                    fire_obj[k] = float(v) if isinstance(v, np.floating) else int(v)
            
            fire_list.append(fire_obj)

        final_json = {
            "project": project_name,
            "location": area_name,
            "latitude": round(float(avg_lat), 5),
            "longitude": round(float(avg_lon), 5),
            "date": date_str,
            "metadata": fire_list
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"      ! Error creating JSON: {e}")
        return False


# --- Helper 5: Direct In-Memory MongoDB Uploader ---
def insert_fires_direct_to_mongodb(gdf, area_name, project_name, date_str):
    """
    Constructs the structured fire metadata payload in memory directly from 
    the GeoDataFrame and inserts it straight into MongoDB without reading from disk.
    """
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "fire_monitoring")
    coll_name = os.getenv("MONGO_COLLECTION_NAME", "detected_fires")

    if not mongo_uri:
        print("     [MONGO SKIP] No MONGO_URI environment variable configured.")
        return

    try:
        avg_lat = gdf.geometry.y.mean()
        avg_lon = gdf.geometry.x.mean()

        fire_list = []
        for _, row in gdf.iterrows():
            raw_time = str(row.get('acq_time', ''))
            formatted_time = raw_time
            if len(raw_time) == 4 and raw_time.isdigit():
                formatted_time = f"{raw_time[:2]}:{raw_time[2:]}"
            elif len(raw_time) == 3 and raw_time.isdigit():
                formatted_time = f"0{raw_time[:1]}:{raw_time[1:]}"

            fire_obj = {
                "latitude": row.get('latitude'),
                "longitude": row.get('longitude'),
                "brightness": row.get('brightness'),
                "acquisition_time": formatted_time,
                "satellite": row.get('satellite'),
                "instrument": row.get('instrument'),
                "confidence": row.get('confidence'),
                "version": row.get('version'),
                "brightness_band_t31": row.get('bright_t31'),
                "fire_radiative_power": row.get('frp'),
                "source": row.get('source_api')
            }
            
            # Convert NumPy types explicitly to native Python primitives for PyMongo/BSON compatibility
            for k, v in fire_obj.items():
                if pd.isna(v) or v is None:
                    fire_obj[k] = None
                elif isinstance(v, (np.integer, np.floating)):
                    fire_obj[k] = float(v) if isinstance(v, np.floating) else int(v)

            fire_list.append(fire_obj)

        # Assemble document directly inside volatile RAM
        document = {
            "project": project_name,
            "location": area_name,
            "latitude": round(float(avg_lat), 5) if not pd.isna(avg_lat) else 0.0,
            "longitude": round(float(avg_lon), 5) if not pd.isna(avg_lon) else 0.0,
            "date": date_str,
            "metadata": fire_list,
            "inserted_at": datetime.utcnow()
        }

        print(f"     [MONGO CONNECT] Connecting to database cluster...")
        with MongoClient(mongo_uri, serverSelectionTimeoutMS=5000) as client:
            db = client[db_name]
            collection = db[coll_name]
            
            print(f"     [MONGO WRITE] Injecting record into '{coll_name}'...")
            result = collection.insert_one(document)
            
            if result.acknowledged:
                print(f"     [MONGO SUCCESS] Inserted successfully! Document ID: {result.inserted_id}")
            else:
                print(f"     [MONGO ERROR] Insertion write unacknowledged by host.")

    except Exception as e:
        print(f"     [MONGO FAILED] Failed to stream data directly to database: {e}")


# ================= JOB FUNCTION  =================
def scan_specific_area(folder_name, project_name, location_id, api_endpoint):
    """
    Runs for a specific folder, using the config details.
    """
    now = datetime.now()
    target_dir = INPUT_SHAPES_FOLDER / folder_name
    
    print(f"\n[{now.strftime('%H:%M:%S')}] >>> Starting job for: {folder_name} (ID: {location_id})")
    
    if not target_dir.exists():
        print(f"   ERROR: Folder not found: {target_dir}")
        return

    all_shapefiles = list(target_dir.rglob("*.shp"))
    shapefiles = [f for f in all_shapefiles if "monitored fires" not in str(f)]

    if not shapefiles:
        print(f"   No .shp files found inside {folder_name}.")
        return

    today = now.date()
    yesterday = today - timedelta(days=1)
    folder_date_label = now.strftime("%d-%m-%Y") 
    json_date_label = now.strftime("%Y-%m-%d")

    for shp_path in shapefiles:
        area_sub_name = shp_path.stem 
        print(f"   > Checking shapefile: {area_sub_name}")
        
        try:
            area_gdf = gpd.read_file(shp_path)
            if area_gdf.crs and area_gdf.crs.to_epsg() != 4326:
                area_gdf = area_gdf.to_crs(epsg=4326)
            
            raw_fires = download_fires_for_geometry(area_gdf, yesterday, today)
            
            # --- SCENARIO 1: No fires in the general region ---
            if raw_fires.empty:
                print("     No fires detected nearby.")
                continue

            points_gdf = gpd.GeoDataFrame(
                raw_fires, 
                geometry=[Point(xy) for xy in zip(raw_fires.longitude, raw_fires.latitude)], 
                crs="EPSG:4326"
            )
            
            clean_gdf = gpd.sjoin(points_gdf, area_gdf, how="inner", predicate="intersects")
            
            # --- SCENARIO 2: Fires found inside your polygon ---
            if not clean_gdf.empty:
                print(f"     ! FOUND {len(clean_gdf)} FIRES. Saving and Syncing...")

                specific_folder_name = f"{folder_date_label}_detected_fires"
                save_dir = OUTPUT_BASE / folder_name / specific_folder_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                base_filename = f"{area_sub_name}_{folder_date_label}"
                
                cols = list(points_gdf.columns)
                clean_gdf[cols].to_file(save_dir / f"{base_filename}.shp")
                pd.DataFrame(clean_gdf[cols].drop(columns='geometry')).to_csv(save_dir / f"{base_filename}.csv", index=False)
                save_geotiff(clean_gdf, save_dir / f"{base_filename}.tif")
                
                # 1. CREATE THE LOCAL JSON FILE (Maintained for your backup)
                save_fire_json(clean_gdf, area_sub_name, project_name, json_date_label, save_dir / "metadata.json")
                
                print(f"     Saved local spatial vector, raster, and metadata reports to: {save_dir}")

                # 2. STREAM DIRECTLY TO MONGODB (In-memory network process)
                insert_fires_direct_to_mongodb(
                    gdf=clean_gdf,
                    area_name=area_sub_name,
                    project_name=project_name,
                    date_str=json_date_label
                )

            # --- SCENARIO 3: Fires nearby, but outside your polygon ---
            else:
                print("     Fires nearby, but NOT inside polygons.")
                
        except Exception as e:
            print(f"     Error processing {area_sub_name}: {e}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] <<< Job complete for '{folder_name}'.")

# ================= DYNAMIC SCHEDULER SETUP =================

print("\n--- Initializing Area Configurations ---")

if not INPUT_SHAPES_FOLDER.exists():
    print(f"CRITICAL ERROR: Main folder not found: {INPUT_SHAPES_FOLDER}")
    exit()

subfolders = [x for x in INPUT_SHAPES_FOLDER.iterdir() if x.is_dir() and x.name != "monitored fires"]

scheduled_count = 0

for folder in subfolders:
    config_path = folder / "config.json"
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            proj_name = config_data.get("project_name", "Unknown Project")
            loc_id = config_data.get("locationId", folder.name) 
            api_url = config_data.get("api_endpoint", None) 
            times = config_data.get("schedule_times", [])
            
            if not times:
                print(f"[!] Warning: 'config.json' in '{folder.name}' has no schedule_times.")
                continue

            api_status = "Linked" if api_url else "No API"
            print(f" • Loaded '{folder.name}' | Project: {proj_name} | API ID: {loc_id} [{api_status}]")
            
            for t in times:
                schedule.every().day.at(t).do(
                    scan_specific_area, 
                    folder_name=folder.name, 
                    project_name=proj_name,
                    location_id=loc_id,
                    api_endpoint=api_url 
                )
                print(f"    -> Scheduled at {t}")
            
            scheduled_count += 1
            
        except Exception as e:
            print(f"[!] Error reading config for '{folder.name}': {e}")
    else:
        print(f"[i] Skipped '{folder.name}' (No config.json found)")

if scheduled_count == 0:
    print("\nWARNING: No valid configurations found! The bot has nothing to do.")
else:
    print(f"\nSUCCESS: {scheduled_count} areas configured. Waiting for scheduled times...")

# =========================================================

while True:
    schedule.run_pending()
    time.sleep(60)

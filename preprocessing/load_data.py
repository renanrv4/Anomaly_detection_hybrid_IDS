from pathlib import Path

# Function to get all .csv files in a dir
def get_files(path):
    path = Path(path)

    return [file for file in path.glob("*.csv") if file.is_file() ]

# ==============================
# Extracting files from ROAD * Use the signal extractions files (they are already decoded)
# ==============================

def get_ambient_files(road_path):
    ambient_path = Path(road_path) / "signal_extractions" / "ambient"

    return get_files(ambient_path)

def get_attack_files(road_path):
    attack_path = Path(road_path) / "signal_extractions" / "attacks"

    return get_files(attack_path)

# Function to map unique ids
def create_id_mapping(ids):
    unique_ids = sorted(set(ids))

    id_to_index = {
        can_id: index
        for index, can_id in enumerate(unique_ids)
    }

    index_to_id = {
        index: can_id
        for can_id, index in id_to_index.items
    }

    return id_to_index, index_to_id
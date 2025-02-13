import json
from datetime import datetime

def write_to_json(data):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    filename = f"logs/{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

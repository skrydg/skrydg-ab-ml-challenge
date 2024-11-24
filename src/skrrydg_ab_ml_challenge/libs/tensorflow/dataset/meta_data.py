import pathlib
import json


class DatasetMetaData:
    def __init__(self, path: pathlib.Path):
        self.path = path

    def write(self, data):
        with open(str(self.path), 'w') as f:
            json.dump(data, f, indent=4)

    def read(self):
        with open(str(self.path), 'r') as f:
            return json.load(f)

    def exists(self):
        return self.path.exists()

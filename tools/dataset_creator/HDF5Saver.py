import h5py
import numpy as np


class HDF5Saver:
    def __init__(self, file_path_to_save="data/carla_dataset.hdf5", groups=None):
        self.file = h5py.File(file_path_to_save, "w")

        # Creating groups to store each type of data
        self.groups = {}
        if groups != None: 
            for group in groups: 
                self.groups[group["id"]] = self.file.create_group(group["id"])
                self.groups[group["id"] + "-transform"] = self.file.create_group(group["id"] + "-transform")
        else: 
            self.groups["rgb"] = self.file.create_group("rgb")
            self.groups["depth"] = self.file.create_group("depth")
            self.groups["semantic_segmentation"] = self.file.create_group("semantic_segmentation")

            # self.groups["ego_speed"] = self.file.create_group("ego_speed")
            # self.groups["bounding_box"] = self.file.create_group("bounding_box")
            # self.groups["bb_vehicles"] = self.groups["bounding_box"].create_group("vehicles")
            # self.groups["bb_walkers"] = self.groups["bounding_box"].create_group("walkers")
        self.groups["timestamps"] = self.file.create_group("timestamps")

    def record_data(self, group_id, data, timestamp):
        timestamp = str(timestamp)
        self.groups[group_id].create_dataset(timestamp, data=data)

    def record_all_timestamps(self, timestamps_list):
        self.groups["timestamps"].create_dataset("timestamps", data=np.array(timestamps_list))

    def close_HDF5(self):
        self.file.close()

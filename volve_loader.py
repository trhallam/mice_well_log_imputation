"""Load Volve Data for input to log_imputer
"""

import pandas as pd


def load_data():
    # The zone map can convert the ZONE log of ints to named Zones.
    zone_map = {
        0: "Seabed",
        1: "NORDLAND",
        2: "Utsira",
        3: "HORDALAND",
        4: "Ty",
        5: "SHETLAND",
        6: "Ekofisk",
        7: "Hod",
        8: "Draupne",
        9: "Heather Shale",
        10: "Heather Sand",
        11: "Hugin C",
        12: "Hugin B3",
        13: "Hugin B2",
        14: "Hugin B1",
        15: "Hugin A",
        16: "Sleipner",
        17: "Skagerrak",
        18: "Smith Bank",
    }

    # Lets rename logs in the HDF5 input to more common terms.
    col_rename_map = {
        "ZONE_NO": "ZONE",
        "DTE": "DT",
        "DTSE": "DTS",
        "DRHOE": "DRHO",
        "GRE": "GR",
        "NPHIE": "NPHI",
        "PEFE": "PEF",
        "RHOBE": "RHOB",
        "RME": "RM",
        "RSE": "RS",
        "RDE": "RD",
        "WELL": "WELL_ID",
    }

    data = pd.read_hdf("data/volve_ml_logs.hdf5").rename(col_rename_map, axis=1)
    # look at deeper zones -> shallow zones poorly sampled/not of interest
    data = data.query("ZONE>=4")
    data["ZONE"] = data["ZONE"].astype(int)
    # data["ZONE_NAME"] = data["ZONE"].map(zone_map)

    test_wells = ["F-4", "F-12", "F-1", "F-15D"]
    train = data[~data.WELL_ID.isin(test_wells)].copy()
    test = data[data.WELL_ID.isin(test_wells)].copy()

    return train, test

import os

import wfdb
import yaml

MITBIH_RECORDS = [
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    121,
    122,
    123,
    124,
    200,
    201,
    202,
    203,
    205,
    207,
    208,
    209,
    210,
    212,
    213,
    214,
    215,
    217,
    219,
    220,
    221,
    222,
    223,
    228,
    230,
    231,
    232,
    233,
    234,
]


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_data(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Downloading {len(MITBIH_RECORDS)} records to '{raw_dir}'...")
    for rec_id in MITBIH_RECORDS:
        rec_name = str(rec_id)
        out_path = os.path.join(raw_dir, rec_name)
        if os.path.exists(out_path + ".dat"):
            continue  # already downloaded
        try:
            wfdb.dl_database("mitdb", dl_dir=raw_dir, records=[rec_name])
        except Exception as e:
            print(f"  Warning: could not download record {rec_id}: {e}")
    print("Download complete.")


def main():
    config = load_config()
    download_data(config["data"]["raw_dir"])


if __name__ == "__main__":
    print("Downloading MIT-BIH Arrhythmia Database...")
    main()

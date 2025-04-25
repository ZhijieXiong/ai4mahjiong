from pymj.botzone.SLDataset1 import SLDataset


for data_mode in ["train", "valid", "test"]:
    for mode in ["Play", "Chi", "Peng", "Gang", "AnGang", "BuGang"]:
        SLDataset.save_data(f"/root/autodl-tmp/mah_jiong_data/{data_mode}",
                            mode,
                            f"/root/code/ai4mahjiong/botzone_data/{data_mode}")


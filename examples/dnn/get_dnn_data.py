from pymj.botzone.SLDataset1 import SLDataset


for data_mode in ["train", "valid", "test"]:
    for mode in ["Play", "Chi", "Peng", "Gang", "AnGang", "BuGang"]:
        print(f"{data_mode}, {mode}")
        SLDataset.save_data(f"/Users/dream/myProjects/Mahjiong/ai4mahjiong/botzone_data/game_data/{data_mode}",
                            mode,
                            f"/Users/dream/myProjects/Mahjiong/ai4mahjiong/botzone_data/sl_data/{data_mode}")


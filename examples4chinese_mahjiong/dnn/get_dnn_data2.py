from pymj.botzone.SLDataset2 import SLDataset


for data_mode in ["train", "test"]:
    for mode in ["Play", "Chi", "Peng", "Gang"]:
        print(f"{data_mode}, {mode}")
        SLDataset.save_data(f"/Users/dream/myProjects/Mahjiong/ai4mahjiong/botzone_data/game_data/{data_mode}",
                            mode,
                            f"/Users/dream/myProjects/Mahjiong/ai4mahjiong/botzone_data/sl_hybrid_data/{data_mode}",
                            10)


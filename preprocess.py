import pandas as pd 
import json
import pickle as pkl


csv_path = "PATH"
save_file = "PATH"
data = pd.read_csv(csv_path)
val_data = data[data["Split"]=="Val"]
train_data = val_data.set_index("File_ID")
data_set = {}

for file_id in list(map(lambda x:x[1:-1], list(train_data.index))):
    cur_info = train_data.loc[f"[{file_id}]"].values.tolist()
    data_set.update({file_id:{"label":cur_info[1:8], "age":cur_info[8], "country":cur_info[-1]}})
with open(save_file, "w") as f:
    json.dump(data_set, f)


data_path = "PATH"
new_path = "PATH"

with open(data_path, "rb") as f:
    data = pkl.load(f)
    new_data = {}
    for key, value in data.items():
        new_data.update({key: value.squeeze()})
    with open(new_path, "wb") as f:
        pkl.dump(new_data, f)


data_path = "PATH"
label_path = "PATH"

with open(data_path, "rb") as f:
    data = pkl.load(f)
    data_keys = data.keys()

    with open(label_path, "r") as f:
        label =json.load(f)
    label_keys = label.keys()
print(len(set(data_keys)))
print(len(set(label_keys)))

from collections import Counter
data_path = r"PATH"
with open(data_path, "rb") as f:
    data = pkl.load(f)
    frame_num = [x.shape[0] for x in data.values()]
    print(sorted(list(Counter(frame_num).items()), key=lambda x:x[0]))
    print(list(data.values())[0].shape)


data_path = r"PATH"
label_path = r"PATH"
with open(data_path, "rb") as f:
    data = pkl.load(f)
    invalid_token = [i[0]for i in data.items() if (i[1]["token"] >= 23672).any()]
    print(invalid_token)
    data_keys = data.keys()
    with open(label_path, "r") as f:
        label =json.load(f)
    label_keys = label.keys()

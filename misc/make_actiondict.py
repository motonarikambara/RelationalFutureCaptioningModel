import json

with open("./annotations/ponnet/captioning_train.json", "r") as f:
    cap_data = json.load(f)
sent_list = []
for i in cap_data:
    sent_list.append(i["sentence"])


action_dict = {}
for i in sent_list:
    splits = i.split()
    action = splits[0]
    if action in action_dict:
        action_dict[action] += 1
    else:
        action_dict[action] = 1

print(action_dict)
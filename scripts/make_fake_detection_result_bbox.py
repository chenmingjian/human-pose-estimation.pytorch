import json 

j = json.load(open("./data/crowdpose/annotations/crowdpose_val.json"))

d = []

template = {
    'bbox': [244.05099093550018,
        170.61324112294565,
        74.56380466590753,
        73.94038239037835],
    'category_id': 1,
    'image_id': 532481,
    'score': 1
}
for i in j['annotations']:
    t = template.copy()
    t['bbox'] = i['bbox']
    t['image_id'] = i['image_id']
    d.append(t)


json.dump(d, open("./data/crowdpose/person_detection_result/val_box.json", 'w'))
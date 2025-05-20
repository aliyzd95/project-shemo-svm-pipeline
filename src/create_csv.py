import json
import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml"))['preprocess']


def preprocess(input_path, output_path):
    label2id = {"anger": 0, "surprise": 1, "happiness": 2, "sadness": 3, "neutral": 4, "fear": 5}

    paths = []
    labels = []
    with open(input_path, encoding='utf-8') as ms:
        modified_shemo = json.loads(ms.read())
        for file in modified_shemo:
            path = modified_shemo[file]["path"]
            label = label2id[modified_shemo[file]["emotion"]]
            if label != 5:
                paths.append(f'../{path}')
                labels.append(label)

    data_dict = {'path': paths, 'label': labels}
    df = pd.DataFrame(data_dict)
    df.to_csv(output_path)




if __name__ == "__main__":
    preprocess(params["input"], params["output"])




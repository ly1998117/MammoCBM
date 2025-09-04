import torch
import numpy as np


class Analysis:
    @staticmethod
    def tsne():
        from utils.tsne import tsne, plot
        emb = torch.load('dataset/PreStudy/Cache/efficientnet-b0-MM-train-PoolingTrue-Fold0.pth', map_location='cpu')
        data_m, label_m = {}, {}
        for d in emb:
            if isinstance(d['data'], dict):
                for m in d['data'].keys():
                    if d['data'][m].ndim > 1:
                        data = torch.nn.functional.adaptive_avg_pool3d(d['data'][m], 1).flatten()
                    else:
                        data = d['data'][m].numpy()
                    if m in data_m:
                        data_m[m].append(data)
                    else:
                        data_m[m] = [data]

                    if m in label_m:
                        label_m[m].append(d['label'])
                    else:
                        label_m[m] = [d['label']]
            else:
                if 'MM' in data_m:
                    data_m['MM'].append(d['data'].numpy())
                    label_m['MM'].append(d['label'])
                else:
                    data_m['MM'] = [d['data'].numpy()]
                    label_m['MM'] = [d['label']]

        for m in data_m.keys():
            emb = np.stack(data_m[m])
            label = np.stack(label_m[m])
            emb = tsne(emb)
            plot(emb, label, f'{m}-tSNE')


if __name__ == "__main__":
    Analysis.tsne()

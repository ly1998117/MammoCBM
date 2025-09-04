import torch
from config import Config
from utils.trainHelper import TrainHelper as _TrainHelper
from dataset import DataModule


class TrainHelper(_TrainHelper):
    def set_datamodule(self):
        self.datamodule = DataModule(self.config)

    def gradcam(self, path, modality=None):
        from utils.grad_cam import GradCAM
        ori_images = {k: d.unsqueeze(0) if isinstance(d, torch.Tensor) else torch.FloatTensor(d).unsqueeze(0) for k, d
                      in self.datamodule.transform['pure'](path).items()}
        path = {k: d.unsqueeze(0) if isinstance(d, torch.Tensor) else torch.FloatTensor(d).unsqueeze(0) for k, d in
                self.transform(path).items()}

        self.load_from_configfile(output_dir=self.config.output_dir, model=self.model)
        logits, concept_logits = self.model(path, modality=modality)
        concept_logits = (concept_logits.sigmoid() > 0.5).cpu().squeeze()

        def getcam(modality, target_layers, dirname, class_idx=None):
            cam = GradCAM(nn_module=self.model, target_layers=target_layers)
            acti_map = cam.compute_map(path, class_idx=class_idx, retain_graph=False, layer_idx=-1, modality=modality)
            heatmap = cam._upsample_and_post_process(acti_map, path[modality])
            cam.plot_slices(ori_images[modality], heatmap,
                            dirpath=f'{self.config.output_dir}/GradCAM/{dirname}/{modality}',
                            filename=f'gradcam',
                            figsize=(256, 512, 256),
                            skip=10)

        for cid in range(concept_logits.shape[-1]):
            for m in ['DWI', 'CE', 'ADC', 'T2WI']:
                getcam(m, f"encoder.encoder.{m}._bn1",
                       f'LONGSHIQUN_42Y_0015431853/concept_{cid}_{concept_logits[cid]}', class_idx=cid)


if __name__ == '__main__':
    config = Config.config(postfix='concept', iter_train=True)
    config.include_concept = True
    print(config.warmup_ratio)
    helper = TrainHelper(config=config)
    # helper.gradcam({'ADC': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/ADC.nii.gz',
    #                 'CE': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/CE.nii.gz',
    #                 'DWI': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/DWI.nii.gz',
    #                 'T2WI': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/T2WI.nii.gz',
    #                 'TIC': [0, 0, 1, 1, 0, 0]},
    #                'MM')
    helper.run()

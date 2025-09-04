import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from chatGPT import ChatGPT, Prompt

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportProcess:
    def __init__(self, data_dir='PreStudy',
                 api_base='https://www.aigptx.top/v1',
                 api_key='sk-XXMH4w8kdd52f1C8Ae79T3BLbkFJ69577711C63a45348247',
                 model='gpt-4o'):
        self.data_dir = f"dataset/{data_dir}/CSV/report"
        self.datalist = f"dataset/{data_dir}/CSV/data_split/datalist.csv"
        self.report = pd.read_csv(f'{self.data_dir}/report.csv')
        self.prompt = Prompt()
        self.gpt = ChatGPT(
            api_base=api_base,
            api_key=api_key,
            model=model
        )

    def to_structure(self):
        # Split the report into sections

        for patient in tqdm(self.report.to_dict(orient='records'), desc='Report Structuring', total=len(self.report)):
            position = patient['position']
            info = f'{patient["pathology"]}-{patient["name_en"]}-{patient["position"]}'
            if os.path.exists(f'{self.data_dir}/report_structure/{patient["pathology"]}/{info}.json'):
                continue
            os.makedirs(f'{self.data_dir}/report_structure/{patient["pathology"]}', exist_ok=True)
            freetext = (f"影像表现:\n {patient['影像表现']} \n "
                        f"影像诊断:\n {patient['影像诊断']} \n "
                        f"病理结果:\n {patient['病理结果']} \n ")
            prompts = self.prompt.freetext2structure(freetext, position=position)
            structure = self.gpt(prompts=prompts).replace('```json', '').replace('```', '')
            with open(f'{self.data_dir}/report_structure/{patient["pathology"]}/{info}.json', 'w') as f:
                json.dump(json.loads(structure), f, indent=4)
            self.report.loc[self.report['name_en'] == patient['name_en'], 'structure'] = structure
            self.report.to_excel(f'{self.data_dir}/report_structure.xlsx', index=False)

    def structure_to_report(self, structure, pred_prob, filepath, diagnose):
        with open(f'{filepath}.json', 'w') as f:
            json.dump(structure, f, indent=4)
        prompts = self.prompt.structure2report(structure, pred_prob, diagnose)
        report = self.gpt(prompts=prompts)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(report)

    @staticmethod
    def dict_key_check(d1, d2) -> bool:
        if isinstance(d1, dict) and isinstance(d2, dict):
            keys_d1_not_in_d2 = set(d1.keys()) - set(d2.keys())
            if keys_d1_not_in_d2:
                for key in keys_d1_not_in_d2:
                    logger.warning(f'Key "{key}" found in d1 but not in d2.')
                return False

            keys_d2_not_in_d1 = set(d2.keys()) - set(d1.keys())
            if keys_d2_not_in_d1:
                for key in keys_d2_not_in_d1:
                    logger.warning(f'Key "{key}" found in d2 but not in d1.')
                return False

            # 递归检查嵌套字典
            for key in d1:
                if isinstance(d1[key], dict):
                    if not isinstance(d2[key], dict):
                        logger.warning(f'Value for key "{key}" is a dict in d1 but not in d2.')
                        return False
                    if not ReportProcess.dict_key_check(d1[key], d2[key]):
                        logger.warning(f'Nested dicts for key "{key}" do not match.')
                        return False
            return True
        elif isinstance(d1, dict) != isinstance(d2, dict):
            logger.warning('One of the inputs is a dict while the other is not.')
            return False
        else:
            # 如果两个都不是字典，则无需比较键
            return True

    def structure_to_concept(self, checked=False):
        pop_keys = [
            'add', "[add: {key1:value1}]", "[add: BI-RADS_category]", "additional_findings", "[add: Pathology]",
            "[add: {pathology_results}]",
        ]
        template = json.loads(self.prompt.template)
        info = pd.read_csv(os.path.join(os.path.dirname(self.data_dir), 'data_split', 'datalist.csv'))[
            ['name', 'path', 'modality', 'pid']].drop_duplicates()
        dir_path = os.path.join(self.data_dir, 'checked_structure' if checked else 'report_structure')
        patients = []
        for pathology in os.listdir(dir_path):
            if not os.path.isdir(os.path.join(dir_path, pathology)):
                continue
            for patient in os.listdir(os.path.join(dir_path, pathology)):
                print(patient)
                position = patient.split('.')[0].split('-')[-1]
                name = '-'.join(patient.split('.')[0].split('-')[1:-1])
                patient = os.path.join(dir_path, pathology, patient)
                bbox = patient.replace('checked_structure' if checked else 'report_structure', 'bbox')
                try:
                    with open(patient) as f:
                        patient = json.load(f)
                        for k in pop_keys:
                            if k in patient:
                                patient.pop(k)
                        if '4' in patient['BI-RADS_category']:
                            v = patient['BI-RADS_category'].pop('4')
                            if v is True:
                                if pathology == 'BC':
                                    patient['BI-RADS_category']['4C'] = v
                                else:
                                    patient['BI-RADS_category']['4A'] = v
                except json.decoder.JSONDecodeError as e:
                    print(pathology, name)
                    raise e

                if not ReportProcess.dict_key_check(template, patient):
                    raise KeyError(f'{pathology}-{name}')
                if os.path.exists(bbox):
                    try:
                        with open(bbox) as f:
                            bbox = json.load(f)
                            for k in pop_keys:
                                if k in bbox:
                                    bbox.pop(k)
                            if '4' in bbox['BI-RADS_category']:
                                bbox['BI-RADS_category'].pop('4')
                    except json.decoder.JSONDecodeError as e:
                        print(pathology, name)
                        raise e
                    if not ReportProcess.dict_key_check(template, bbox):
                        raise KeyError(f'{pathology}-{name}')
                else:
                    bbox = None

                patients.append({
                    'name': name,
                    'pathology': pathology,
                    'position': position,
                    'concept': patient,
                    'bbox': bbox
                })
        patients = pd.DataFrame(patients)
        print(patients.loc[~patients['name'].isin(info['name'])])
        print(info.loc[~info['name'].isin(patients['name'])])
        patients = patients.merge(info, on='name')
        patients.to_csv(f'{self.data_dir}/concepts.csv', index=False)

    def _extract_tic(self, datalist):
        def _get_structure(img_path, reverse=True):
            if reverse:
                img_path = img_path.replace(img_path.split('/')[0], f"{img_path.split('/')[0]}_TIC")
            prompts = self.prompt.tic_points('dataset/' + img_path)
            structure = self.gpt(prompts=prompts).replace('```json', '').replace('```', '')
            return structure

        results = []
        errors = []
        for patient in tqdm(datalist, desc='TIC Extraction', total=len(datalist)):
            img_path = literal_eval(patient['path'])['TIC']
            i = 0
            while i < 4:
                try:
                    structure = _get_structure(img_path)
                    structure = json.loads(structure)
                    break
                except json.JSONDecodeError:
                    print(f'JSONDecodeError: {img_path}')
                    structure = _get_structure(img_path, reverse=i < 2)
                    i += 1
            patient['tic'] = structure
            print(f'{img_path} TIC: {structure}')
            if i == 4 or 'error' in structure:
                errors.append(patient)
            else:
                results.append(patient)
        results = pd.DataFrame(results)
        errors = pd.DataFrame(errors)
        return results, errors

    def extract_tic(self):
        if os.path.exists(f'{self.data_dir}/tic_error.csv'):
            datalist = pd.read_csv(f'{self.data_dir}/tic_error.csv').to_dict(orient='records')
            results, errors = self._extract_tic(datalist)
            errors = pd.concat([results, errors], axis=0)
            errors.to_csv(f'{self.data_dir}/tic_error.csv', index=False)
        else:
            datalist = pd.read_csv(self.datalist).to_dict(orient='records')
            results, errors = self._extract_tic(datalist)
            results.to_csv(f'{self.data_dir}/tic.csv', index=False)
            errors.to_csv(f'{self.data_dir}/tic_error.csv', index=False)

    def tic_compute(self):
        datalist = pd.read_csv(f'{self.data_dir}/tic.csv')
        if os.path.exists(f'{self.data_dir}/tic_error.csv'):
            error = pd.read_csv(f'{self.data_dir}/tic_error.csv')
            if error['name'].isin(datalist['name']).sum() == 0:
                error['tic'] = error['tic'].map(lambda x: str(literal_eval(x.replace('*60', ''))))
                datalist = pd.concat([error, datalist])
        datalist = datalist.to_dict(orient='records')
        results = []
        for patient in tqdm(datalist, desc='TIC Extraction', total=len(datalist)):
            tic = literal_eval(patient['tic'])
            start, increase, middle, end, peak = tic['start'], tic['increase'], tic['middle'], tic['end'], tic['peak']
            if abs(peak[0] - end[0]) < abs(peak[0] - middle[0]):
                # peak = (increase[0] + middle[0]) / 2, (increase[1] + middle[1]) / 2
                peak = increase
            earlyR = 100 * (increase[1] - start[1]) / (start[1] + 1e-2)
            lateR = 100 * (end[1] - peak[1]) / (peak[1] + 1e-3)
            patient['earlyR'] = f'{earlyR:.2f}'
            patient['lateR'] = f'{lateR:.2f}'
            if earlyR < 50:
                patient['Initial enhancement phase'] = "Slow"
            elif earlyR <= 100:
                patient['Initial enhancement phase'] = "Medium"
            else:
                patient['Initial enhancement phase'] = "Fast"
            if lateR <= -10:
                patient['Delayed phase'] = "Wash-out"
            elif lateR < 10:
                patient['Delayed phase'] = "Plateau"
            else:
                patient['Delayed phase'] = "Persistent"
            results.append(patient)
        results = pd.DataFrame(results)
        results.to_csv(f'{self.data_dir}/tic.csv', index=False)

    def tic_metrics(self):
        gt = self.tic_gt()
        pred = pd.read_csv(f'{self.data_dir}/tic.csv')
        pred = pred.merge(gt, on=['pid', 'name'], suffixes=(' pred', ' gt')).reset_index(drop=True)
        pred.to_csv(f'{self.data_dir}/tic_merge.csv', index=False)
        assert len(gt) == len(pred), f'{len(gt)} {len(pred)}'
        acc1 = sum(pred['Initial enhancement phase gt'] == pred['Initial enhancement phase pred']) / len(pred)
        acc2 = sum(pred['Delayed phase gt'] == pred['Delayed phase pred']) / len(pred)
        print(f'Initial enhancement phase: {acc1:.4f}')
        print(f'Delayed phase: {acc2:.4f}')
        pd.DataFrame([{'Initial enhancement phase': acc1, 'Delayed phase': acc2}]).to_csv(
            f'{self.data_dir}/tic_acc.csv', index=False
        )

    def tic_gt(self):
        gt = pd.read_csv(f'{self.data_dir}/TIC-GT.csv')
        gt['Initial enhancement phase'] = gt['Initial enhancement phase'].map(str.capitalize)
        gt['Delayed phase'] = gt['Delayed phase'].map(str.capitalize)
        gt.to_csv(f'{self.data_dir}/tic_gt.csv', index=False)
        return gt[['pid', 'name', 'Initial enhancement phase', 'Delayed phase']].reset_index(drop=True)


def tic(data_dir):
    rp = ReportProcess(
        data_dir=data_dir,
        api_base='https://c-z0-api-01.hash070.com/v1',
        api_key='sk-E3LDC8hj6e07fd77986aT3BLbKFJCd97ac6fA09E452794cc',
        model='gemini-2.5-flash')
    # rp.extract_tic()
    rp.tic_compute()
    rp.tic_metrics()


def reportGeneration(dirname='PreStudy2'):
    rp = ReportProcess(data_dir=dirname,
                       api_base='https://c-z0-api-01.hash070.com/v1',
                       api_key='sk-E3LDC8hj6e07fd77986aT3BLbKFJCd97ac6fA09E452794cc',
                       model='chatgpt-4o-latest')
    rp.to_structure()
    rp.structure_to_concept(checked=True)
    # rp.tic_right()


if __name__ == "__main__":
    # tic('ProspectiveData')
    reportGeneration('ProspectiveData')

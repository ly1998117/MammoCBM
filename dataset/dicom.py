import os
import pandas as pd
import pydicom as dicom
import shutil

def read_info(dir_path):
    infos = []
    for p in os.listdir(dir_path):
        if 'DS_Store' in p:
            continue
        pp = os.path.join(dir_path, p)
        for m in os.listdir(pp):
            if 'DS_Store' in m:
                continue
            mp = os.path.join(pp, m)
            for f in os.listdir(mp):
                fp = os.path.join(mp, f)
                ds = dicom.dcmread(fp)
                patient_name = ds.get("PatientName", "Unknown")
                patient_id = ds.get("PatientID", "Unknown")
                patient_age = ds.get("PatientAge", "Unknown")

                if patient_name == "Unknown":
                    raise KeyError(fp)
                # Print the extracted metadata for each DICOM file
                print(f"File: {fp}")
                print(f"Patient Name: {patient_name}")
                print(f"Patient Age: {patient_age}")
                print(f"Patient ID: {patient_id}")
                print("-" * 50)  # Separator line for readability
                infos.append({
                    'dirpath': dir_path,
                    'dirname': p,
                    'id': patient_id,
                    'name': patient_name,
                    'age': patient_age
                })
                break
    return pd.DataFrame(infos)


def remove(df):
    for i, row in df.iterrows():
        dirpath = row['dirpath']
        dirname = row['dirname']
        o_path = os.path.join(dirpath, dirname)
        new_path= os.path.join(dirpath, f'{row['name']}_{row['age']}_{row['id']}')
        try:
            shutil.move(o_path, new_path)
        except:
            pass

if __name__ == "__main__":
    dir_path = '/Users/liuyang/Downloads/CLIP_sec2_benign_rename'
    df = read_info(dir_path)
    # remove(df)

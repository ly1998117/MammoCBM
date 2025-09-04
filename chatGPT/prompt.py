import base64
import json


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class Prompt:
    def __init__(self, filepath='chatGPT/BIRADS.json'):
        self.template = None
        with open(filepath, 'r') as f:
            if filepath.endswith('.txt'):
                self.template = f.readlines()
            elif filepath.endswith('.json'):
                self.template = f.read()
        with open('chatGPT/TIC_Prompt.txt', 'r') as f:
            self.tic = f.read()

    def freetext2structure(self, freetext, position, language='en'):
        prompt = dict(cn=(
            f"请将以下乳腺癌诊断的自由文本报告转换为标准化的 BI-RADS 结构化报告格式，并输出为 JSON 格式。请确保报告中包括所有相关的医学影像学特征，"
            f"并按照 BI-RADS 指南进行分类和描述。如果原文中缺少某些信息，请在输出的JSON中将相应值设置为null。\n"
            f"要求的 JSON 结构化报告格式如下：\n"
            f"'''{self.template}'''\n"
            f"请确保遵循以上格式，并提供详尽、精确的信息以符合 BI-RADS 分级标准。"),
            en=(
                "Please convert the following free-text breast cancer diagnostic report into a structured BI-RADS report format and output it as JSON. "
                "Ensure that all relevant medical imaging features are included and categorized according to the BI-RADS guidelines. "
                "If certain information is missing in the original text, set the corresponding value in the JSON output to false.\n"
                "The required JSON structured report format should include the following fields:\n"
                f"'''{self.template}'''\n"
                "Ensure you follow the above format and provide detailed, accurate information to comply with the BI-RADS classification standards. "
                "The output JSON should accurately reflect all medical information contained in the input report. "
                # "If the free-text meets the BIRADS criteria, but is not included in the template. Insert in the format [add: {key1:value1}]."
            ))
        prompts = [
            {"role": "system", "content": prompt[language]},
            {"role": "user",
             "content": f"Consider only the free-text report content of {position} breast, ignoring the other side: \n '''{freetext}''' \n"}
        ]
        return prompts

    def tic_points(self, img_path):
        prompts = [
            {"role": "system", "content": self.tic},
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Identify the TIC points in the image below and only provide JSON format."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
                ]
            }
        ]
        return prompts

    def structure2report(self, birads, probability, diagnose):
        with open('chatGPT/Report_Prompt_REVISED.md', 'r') as f:
            reportPrompt = f.read()
        prompts = [
            {"role": "system", "content": reportPrompt},
            {"role": "user",
             "content": f"This is structured data: {birads} \n Malignant probability: {probability} Model Prediction: {diagnose}."}
        ]
        return prompts


if __name__ == "__main__":
    prompt = Prompt('BIRADS.json')
    print(prompt.freetext2structure('dsadasdasd'))

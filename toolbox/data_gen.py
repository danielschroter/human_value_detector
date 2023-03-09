import pandas as pd
import os
import io
import re
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from torchmetrics import AUROC, F1Score

import numpy as np


def get_text_total(lines):
    txt = ' '.join(lines)
    txt = txt.replace("\n", "")
    return txt


def get_id(name):
    result = re.search('article(.*).txt', name)
    return result.group(1)


def get_labels(path):
    with open(path, newline='', encoding="utf-8") as labels:
        label_reader = csv.reader(labels, delimiter='\t', quoting=csv.QUOTE_NONE)
        # labels = [{k: v} for [k, v] in label_reader]
        print(label_reader)
        l_list = []
        for l in label_reader:
            l_list.append(l)

    return l_list


def gen_data_frame(path, col_names, text_gen=False):
    with open(path, newline='', encoding="utf-8") as paragraphs:
        text_reader = csv.reader(paragraphs, delimiter='\t', quoting=csv.QUOTE_NONE)
        # labels = [{k: v} for [k, v] in label_reader]
        para_list = []
        count = 0
        for p in text_reader:
            if text_gen:
                text = p[3] + " " + p[2] + " " + p[1]
                p.append(text)
            else:
                tmp = p[1:]
                aux = []
                for i in range(len(p)):
                    try:
                        p[i] = int(p[i])
                    except:
                        continue
                    if p[i] == 1:
                        aux.append(col_names[i])
                p.append(aux)
            para_list.append(p)
            count += 1
    para_list = para_list[1:]
    df_para = pd.DataFrame(para_list, columns=col_names)
    return df_para


def create_data_file(path_to_template_file, path_to_labels, output_path=None, drop_duplicates=True):
    df_para = gen_data_frame(path_to_template_file,
                             col_names=['Argument ID', 'Conclusion', 'Stance', 'Premise', 'text'], text_gen=True)
    df_labels = gen_data_frame(path_to_labels,
                               col_names=['Argument ID', 'Self-direction: thought', 'Self-direction: action',
                                          'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance',
                                          'Power: resources', 'Face', 'Security: personal', 'Security: societal',
                                          'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility',
                                          'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern',
                                          'Universalism: nature', 'Universalism: tolerance',
                                          'Universalism: objectivity', 'category'])
    new_order = ['Argument ID', 'category', 'Self-direction: thought', 'Self-direction: action',
                 'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance',
                 'Power: resources', 'Face', 'Security: personal', 'Security: societal',
                 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility',
                 'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern',
                 'Universalism: nature', 'Universalism: tolerance',
                 'Universalism: objectivity']
    df_labels=df_labels[new_order]

    res = pd.merge(df_para, df_labels, on=["Argument ID"])


    if drop_duplicates:
        res_no_dup = res.drop_duplicates(subset=["text"])

    # Drop Column that represents "no labels"
    if output_path:
        res_no_dup.to_csv(output_path)
    return res_no_dup


def get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls, power=1):
    weights_for_samples = 1.0/np.array(np.power(samples_per_cls,power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples




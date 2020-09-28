# -*- coding: UTF-8 -*-

import os
import re
import numpy as np
import jieba
import time
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
from gensim.models import KeyedVectors
from sklearn import svm, naive_bayes, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE  # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
from sklearn import metrics


def read_data(dir_path, to_delete_files=[], useless_infos=[], is_Q1=True):
    """
    将文件夹中的文件内容聚集成一个list，list中的一个元素就是一个文件中的内容
    :param dir_path: str, 要读取数据的文件夹路径
    :param to_delete_files: list, 需要删除的文件
    :param useless_infos: list, 无用信息，如姓名、陈述者签名之类
    :return: 读入的文件内容、文件名
    """
    file_names = os.listdir(dir_path)
    file_names.sort()
    read_infos = []
    read_file_names = []
    for i in range(len(file_names)):
        if file_names[i] in to_delete_files:
            continue
        cur_file_path = os.path.join(dir_path, file_names[i])
        fr = open(cur_file_path, mode='r', encoding='utf-8')
        cur_file_str = ''
        flag = False
        for line in fr:
            if is_Q1 and (line == "最后诊断：\n" or line == "初步诊断：\n" or line == "入院诊断：\n" or line == "入院诊断:\n"):  # 第1问只保留前两段数据
                break
            if not is_Q1 and '住院天数' in line:
                flag = True
            # 去除字符串中的空格、制表符，某些特殊字符，并把英文的逗号、冒号转成中文的逗号、冒号
            line = line.replace(' ', '').replace('\t', '').replace('\ufeff', '').replace('\u3000', '').\
                replace(',', '，').replace(':', '：')
            if not line.split():  # 跳过空行
                continue
            if line.split('：')[0] in useless_infos:
                continue
            cur_file_str += line
        if not flag and not is_Q1:  # 第2问，并且当前文件不包含住院天数属性
            continue
        read_file_names.append(file_names[i])
        read_infos.append(cur_file_str)
        fr.close()
    return read_infos, read_file_names


def write_data(file_infos, save_file_names, dir_path):
    """
    :param file_infos: list, 文件内容组成的字符串列表
    :param save_file_names: list, 保存文件的名字列表
    :param dir_path: str，保存路径
    :return: None
    """
    for file_info, file_name in zip(file_infos, save_file_names):
        save_file_path = os.path.join(dir_path, file_name)
        fw = open(save_file_path, mode='w', encoding='utf-8')
        fw.write(file_info)
        fw.close()


def write_segmented_file(file_infos, seg_file_save_path):
    """
    分割file_infos中的字符串并保存
    :param file_infos: list, 文件内容组成的字符串列表
    :param seg_file_save_path: 分词文件保存路径
    :return: None
    """
    jieba.load_userdict('userdict.txt')
    fw = open(seg_file_save_path, mode='w', encoding='utf-8')
    for line in file_infos:
        new_line = ''
        sub_line_list = re.split('，|。|！|？|、|：|\n|\t| |《|》|‘|’|“|”|（|）|"|:|\(|\)|;|；|\.|', line)
        for sub_line in sub_line_list:
            if sub_line != '':
                new_line = new_line + (' '.join(jieba.cut(sub_line))) + ' '
        fw.write(new_line + '\n')
    fw.close()


def get_tfidf_words_and_weight_matrix(seg_file_path):
    fr = open(seg_file_path, mode='r', encoding='utf-8')
    corpus = fr.readlines()
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    weight_matrix = tfidf.toarray()
    words = vectorizer.get_feature_names()
    return words, weight_matrix


def get_samples_with_specific_label(X, Y, specific_labels=[0.0, 1.0, 4.0]):
    counter = Counter(Y)
    length = 0
    for label in specific_labels:
        length += counter[label]
    sub_X = np.zeros(shape=(length, len(X[0])))
    sub_Y = np.zeros(length)
    cnt = 0
    for x, y in zip(X, Y):
        if y in specific_labels:
            sub_X[cnt] = copy.deepcopy(x)
            sub_Y[cnt] = copy.deepcopy(y)
            cnt += 1
    return sub_X, sub_Y


def get_Q1_labels(dir_path):
    file_names = os.listdir(dir_path)
    file_names.sort()
    labels = np.zeros(len(file_names))
    disease_name_to_label = {'肾病': 0, '酮症': 1, '心脏病': 2, '眼病': 3, '周围神经病': 4, '足病': 5}
    for i in range(len(file_names)):
        labels[i] = disease_name_to_label[str(file_names[i].split('_')[0])]
    return labels


def get_Q2_labels(dir_path='Q2/data_0'):
    file_names = os.listdir(dir_path)
    file_names.sort()
    day_re = re.compile(r'[\d]+')
    labels = np.zeros(len(file_names))
    for i in range(len(file_names)):
        fr = open(os.path.join(dir_path, file_names[i]), mode='r', encoding='utf-8')
        for line in fr:
            if '住院天数' in line:
                labels[i] = int(day_re.search(line).group())
    return labels


def read_format_csv_file(info_number, format_info_file='Q1/format_info.csv'):
    fr = open(format_info_file, mode='r', encoding='utf-8')  # 读入basic_info
    begin_idx = 1
    title = fr.readline().replace('\n', '').split(',')[begin_idx:]
    format_info = np.zeros(shape=(info_number, len(title)), dtype=np.float)
    for i in range(info_number):
        line = fr.readline()
        sub_line_list = line.replace('\n', '').split(',')[begin_idx:]
        for j in range(len(sub_line_list)):
            if sub_line_list[j] != '':
                format_info[i][j] = float(sub_line_list[j])
    return format_info


def get_X(words, word_weight_matrix, legal_words):
    """
    :param words: numpy array, jieba分词得到的所有词汇
    :param weight_matrix: 每个句子中的每个词汇对应的tf-idf权重
    :return: 文本对应的向量
    """
    key_words_index_list = []
    key_words_re = re.compile(r'[\u4e00-\u9fa5]+')  # 简单过滤掉数字和字母
    for i in range(len(words)):
        if key_words_re.search(words[i]) and words[i] in legal_words:
            key_words_index_list.append(i)

    format_info = read_format_csv_file(len(word_weight_matrix))
    word_vectors = np.zeros(shape=(len(word_weight_matrix), len(key_words_index_list)))
    for i in range(len(word_weight_matrix)):
        for j in range(len(key_words_index_list)):
            word_vectors[i][j] = word_weight_matrix[i][key_words_index_list[j]]

    min_max_scaler = preprocessing.MinMaxScaler()
    word_vectors = min_max_scaler.fit_transform(word_vectors)
    format_info = min_max_scaler.fit_transform(format_info)
    X = np.hstack((format_info, word_vectors))
    return X


def get_sub_file_infos(X, Y, specific_labels):
    counter = Counter(Y)
    length = 0
    for label in specific_labels:
        length += counter[label]
    sub_X = []
    sub_Y = []
    for x, y in zip(X, Y):
        if y in specific_labels:
            print('a')
            copy_x = copy.deepcopy(x)
            copy_y = copy.deepcopy(y)
            sub_X.append(copy_x)
            sub_Y.append(copy_y)
    return sub_X, sub_Y


def generate_training_samples(file_infos, label, generate_dir_path, generate_number=100):
    """
    用于生成样例
    :param init_file_path: str, 初始数据文件路径
    :param generate_file_path: str, 生成数据保存路径
    :return: None
    """
    print(len(file_infos))
    sentences_pool = []
    sentence_length_cnt = 0
    for line in file_infos:
        sub_line_list = re.split('。|\n|，', line)
        for sub_line in sub_line_list:
            if sub_line != '':
                sentences_pool.append(sub_line)
                sentence_length_cnt += 1
    avg_length = sentence_length_cnt//(len(file_infos))
    sentences_pool_length = len(sentences_pool)
    generate_samples = []
    generate_labels = []
    for i in range(generate_number):
        new_line = ''
        for j in range(avg_length):
            idx = np.random.randint(0, sentences_pool_length)
            new_line += (sentences_pool[idx]+'。')
        generate_samples.append(new_line)
        generate_labels.append(label)
    return generate_samples, generate_labels


def write_format_csv_file(dir_path='Q1/data_0', format_info_save_file='Q1/format_info.csv', padding=False):
    """
    利用正则表达式匹配，筛选出文本中的数据信息。数值默认为-1

    眼病关键特征：
      病史超过10年
      I期：微血管瘤，小出血点
      II期：硬性渗出
      III期：棉絮状软性渗出
      IV期：新生血管形成、玻璃体积血
      V期：纤维血管增值、玻璃体机化
      VI期：牵拉性视网膜脱离、失明
      I~III期：非增值期视网膜病变(NPDR)
      IV~VI期：增殖期视网膜病变(PDR)
    足病关键特征：溃疡、感染
    酮症关键特征：心跳过快;低血压;呼出气体呈丙酮味，换言之，如无异常呼出气味，大概率不是酮症

    肾病关键特征：
      1/2型糖尿病，病史超过10年，
      III期：开始出现微量白蛋白尿，UAER：20~200μg/min
      IV期：UAER：200μg/min以上，尿蛋白总量>0.5g/24h，伴有水肿和高血压，肾功能逐渐减退
      V期：尿毒症，UAER降低，血肌酐升高，血压升高，白蛋白/肌酐比率30, 30~299, >= 300

    周围神经病变关键特征：
      远端对称性多发性神经病变：手足远端感觉异常，伴痛觉过敏、疼痛，后期感觉丧失，
      局灶性单神经病变：起病急，病变神经分布区域疼痛
      ...
    :param dirpath: 保存简单处理过的文件的文件夹的路径
    :param basic_info_save_file: 保存基本信息的csv文件，包括 年龄、性别、体温、呼吸、血压(高压/低压)、脉搏等
    :return: None
    """
    data_0_files_name = os.listdir(dir_path)
    data_0_files_name.sort()
    info_features = ['id', '文件名', '并发症', '糖尿病类型', '年龄', '性别', '体温', '脉搏', '呼吸', '血压-高压', '血压-低压',
                           '空腹血糖', '随机血糖', '渗透压', '食欲较差', '血肌酐(umol/L)', '糖化血红蛋白(%)', '尿微量白蛋白/尿肌酐',
                           '尿白细胞', '尿白蛋白(mg/l)', '尿葡萄糖', '尿蛋白', '水肿', '尿氮素', '尿素', '病史(年)']
    init_values = ['-1' for fea in info_features]
    disease_name_to_label = {'肾病': '0', '酮症': '1', '心脏病': '2', '眼病': '3', '周围神经病': '4', '足病': '5'}
    gender_to_number = {'女': '0', '男': '1'}
    ver1_re = re.compile(r'[1Ii]型?糖尿病?')
    ver2_re = re.compile(r'((II)|2|(ii))型?糖尿病?')
    gender_re = re.compile(r'性别[：:]?[男女]')
    temperature_re = re.compile(r'体温[：:]?\d+\.?\d*')
    pulse_re = re.compile(r'脉搏[：:]?\d+\.?\d*')
    breathing_re = re.compile(r'呼吸[：:]?\d+\.?\d*')
    blood_pressure_re = re.compile(r'血压[：:]?(\d+\.?\d*)/(\d+\.?\d*)')
    age_re = re.compile(r'年龄[：:]?\d+')
    fpg_re = re.compile(r'空腹(血糖)?[：:]?[(\u4e00-\u9fa5)（）]*(\d+\.?\d*)[-]?(\d+\.?\d*)?(mmol/)[Ll]')
    rpg_re = re.compile(r'((((入院)|(随机)|(餐后))(微量)?(血糖)?)|(血糖))[：:]?[(\u4e00-\u9fa5)（）]*(\d+\.?\d*)[-]?(\d+\.?\d*)?(mmol/)[Ll]')
    osmotic_pre_re = re.compile(r'渗透压[：:]?(\d+\.?\d*)')
    appetite_re = re.compile(r'[，。、：][(\u4e00-\u9fa5)、]*(食欲)(食量)?((减退)|(较?差)|(下降)|(很差)|(良?好)|(一般)|(旺盛)|(正常)|(尚可)|(增多))')
    creatinine_re = re.compile(r'血?肌酐[：:]?[(\u4e00-\u9fa5)（）]*(\d+\.?\d*)[μmu]mol/[Ll]')
    ghb_re = re.compile(r'((糖化血红蛋白)|([Gg][Hh][Bb]))[：:]?(\d+\.?\d*)%')
    acr_re = re.compile(r'((尿微量白蛋白/尿肌酐)|(ACR))[：:]?(\d+\.?\d*)mg/g')
    leu_re = re.compile(r'((尿白细胞(（高倍镜）)?)|(LEU))[：:]?((\d+\.?\d*)|[+])[+]*[-]?')
    alb_re = re.compile(r'((尿?(微量)?白蛋白)|([Aa][Ll][Bb]))(\d+\.?\d*)m?g/[Ll]')
    glu_re = re.compile(r'((尿葡萄糖)|([Gg][Ll][Uu]))[：:]?(\d*\.?\d*)[+]+[-]?')
    pro_re = re.compile(r'((尿蛋白)|([Pp][Rr][Oo]))[：:]?(\d*\.?\d*)[+]+[-]?')
    edema_re = re.compile(r'[，。、：][(\u4e00-\u9fa5)、]*水肿')
    number_re = re.compile(r'\d+\.?\d*')
    disease_year_re = re.compile(r'主诉[:：]?[(\u4e00-\u9fa5)]*[\d]+[(\u4e00-\u9fa5)]*[，。]')
    format_feature_list = ['id', '糖尿病类型', '年龄', '体温', '脉搏', '血压-高压', '血压-低压',
                           '空腹血糖', '随机血糖', '渗透压', '食欲较差', '血肌酐(umol/L)', '糖化血红蛋白(%)',
                           '尿白细胞', '尿白蛋白(mg/l)', '尿葡萄糖', '尿蛋白', '水肿', '病史(年)']
    format_basic_info = np.zeros(shape=(len(data_0_files_name), len(format_feature_list)), dtype=np.float)
    # 将数据整理成格式数据
    for i in range(len(data_0_files_name)):
        fr = open(os.path.join(dir_path, data_0_files_name[i]), mode='r', encoding='utf-8')
        cur_basic_info = dict(zip(info_features, init_values))
        cur_basic_info['id'] = str(i)
        cur_basic_info['文件名'] = data_0_files_name[i]
        cur_label = disease_name_to_label[data_0_files_name[i].split('_')[0]] if data_0_files_name[i].split('_')[0] in disease_name_to_label else '-1'
        cur_basic_info['并发症'] = str(cur_label)
        for line in fr:
            ver1_obj = ver1_re.search(line)
            ver2_obj = ver2_re.search(line)
            gender_info_obj = gender_re.search(line)
            temperature_info_obj = temperature_re.search(line)
            pulse_info_obj = pulse_re.search(line)
            breathing_info_obj = breathing_re.search(line)
            blood_pressure_info_obj = blood_pressure_re.search(line)
            age_obj = age_re.search(line)
            fpg_obj = fpg_re.search(line)
            rpg_obj = rpg_re.search(line)
            osmotic_pre_obj = osmotic_pre_re.search(line)
            appetite_obj = appetite_re.search(line)
            creatinine_obj = creatinine_re.search(line)
            ghb_obj = ghb_re.search(line)
            acr_obj = acr_re.search(line)
            leu_obj = leu_re.search(line)
            alb_obj = alb_re.search(line)
            glu_obj = glu_re.search(line)
            pro_obj = pro_re.search(line)
            edema_obj = edema_re.search(line)
            disease_year_obj = disease_year_re.search(line)
            if ver1_obj:
                cur_basic_info['糖尿病类型'] = '1'
            elif ver2_obj:
                cur_basic_info['糖尿病类型'] = '2'
            if gender_info_obj:
                sub_str = gender_info_obj.group()
                cur_basic_info['性别'] = gender_to_number[sub_str[3:]] if ':' in sub_str or '：' in sub_str else \
                    gender_to_number[sub_str[2:]]
            if temperature_info_obj:
                sub_str = temperature_info_obj.group()
                cur_basic_info['体温'] = number_re.search(sub_str).group()
            if pulse_info_obj:
                sub_str = pulse_info_obj.group()
                cur_basic_info['脉搏'] = number_re.search(sub_str).group()
            if breathing_info_obj:
                sub_str = breathing_info_obj.group()
                cur_basic_info['呼吸'] = number_re.search(sub_str).group()
            if blood_pressure_info_obj:
                sub_str = blood_pressure_info_obj.group()
                pressure_list = sub_str[3:].split('/') if ':' in sub_str or '：' in sub_str else sub_str[2:].split('/')
                cur_basic_info['血压-高压'] = pressure_list[0]
                cur_basic_info['血压-低压'] = pressure_list[1]
            if age_obj:
                sub_str = age_obj.group()
                cur_basic_info['年龄'] = number_re.search(sub_str).group()
            if fpg_obj:
                sub_str = fpg_obj.group()
                fpg_list = [float(d) for d in number_re.findall(sub_str)]
                cur_basic_info['空腹血糖'] = str(sum(fpg_list)/len(fpg_list))
            if rpg_obj:
                sub_str = rpg_obj.group()
                rpg_list = [float(d) for d in number_re.findall(sub_str)]
                cur_basic_info['随机血糖'] = str(sum(rpg_list)/len(rpg_list))
            if osmotic_pre_obj:
                sub_str = osmotic_pre_obj.group()
                cur_basic_info['渗透压'] = number_re.search(sub_str).group()
            if appetite_obj:  # (食欲)(食量)?((减退)|(较?差)|(下降)|(很差)|(良?好)|(一般)|(旺盛)|(正常)|(尚可)|(增多))
                sub_str = appetite_obj.group()
                bad_appetite = re.compile(r'(食欲)(食量)?((减退)|(较?差)|(下降)|(很差))')
                if '无' not in sub_str and bad_appetite.search(sub_str):
                    cur_basic_info['食欲较差'] = 1  # 食欲较差
                else:
                    cur_basic_info['食欲较差'] = 0
            if creatinine_obj:
                sub_str = creatinine_obj.group()
                if '尿' in sub_str:
                    continue
                num = float(number_re.search(sub_str).group())
                cur_basic_info['血肌酐(umol/L)'] = str(num) if 'mmol' not in sub_str else str(num*1000)
            if ghb_obj:
                sub_str = ghb_obj.group()
                cur_basic_info['糖化血红蛋白(%)'] = number_re.search(sub_str).group()
            if acr_obj:
                sub_str = acr_obj.group()
                cur_basic_info['尿微量白蛋白/尿肌酐'] = number_re.search(sub_str).group()
            if leu_obj:
                sub_str = leu_obj.group()
                if '+' not in sub_str:
                    num = float(number_re.search(sub_str).group())
                    if num > 5 and num <= 10:
                        level = 1
                    elif num > 10 and num <= 15:
                        level = 2
                    elif num > 15 and num < 20:
                        level = 3
                    else:
                        level = 4
                elif '-' in sub_str:
                    level = 0
                else:
                    if number_re.search(sub_str):
                        level = number_re.search(sub_str).group()
                    else:
                        level = sub_str.count('+')
                cur_basic_info['尿白细胞'] = str(level)
            if alb_obj:
                sub_str = alb_obj.group()
                num = float(number_re.search(sub_str).group())
                # print(num, sub_str)
                if 'mg' in sub_str:
                    cur_basic_info['尿白蛋白(mg/l)'] = str(num)
                else:
                    cur_basic_info['尿白蛋白(mg/l)'] = str(num * 1000)
            if glu_obj:
                sub_str = glu_obj.group()
                if number_re.search(sub_str):
                    level = number_re.search(sub_str).group()
                else:
                    level = sub_str.count('+')
                cur_basic_info['尿葡萄糖'] = str(level)
            if pro_obj:
                sub_str = pro_obj.group()
                # print(sub_str)
                if number_re.search(sub_str):
                    level = number_re.search(sub_str).group()
                else:
                    level = sub_str.count('+')
                cur_basic_info['尿蛋白'] = str(level)
            if edema_obj:
                sub_str = edema_obj.group()
                cur_basic_info['水肿'] = '1' if '无' not in sub_str else '0'
            if disease_year_obj:
                sub_str = disease_year_obj.group()
                cur_basic_info['病史(年)'] = number_re.search(sub_str).group()

        for j in range(len(format_feature_list)):
            # print(format_feature_list[j], cur_basic_info[format_feature_list[j]])
            format_basic_info[i][j] = float(cur_basic_info[format_feature_list[j]])
        fr.close()

    if padding:  # 补全数据
        for i in range(format_basic_info.shape[1]):  # 对每一列，用不为-1的数据的中位值补全缺失数据
            mid = np.median(format_basic_info[:, i][format_basic_info[:, i] != -1])
            for j in range(format_basic_info[:, i].shape[0]):
                if format_basic_info[j][i] == -1:
                    format_basic_info[j][i] = mid

    # 写数据
    fw = open(format_info_save_file, mode='w', encoding='utf-8')
    fw.write(','.join(format_feature_list)+"\n")
    for i in range(format_basic_info.shape[0]):
        for j in range(format_basic_info.shape[1]):
            fw.write(str(format_basic_info[i][j]) + ',')
        fw.write('\n')
    fw.close()

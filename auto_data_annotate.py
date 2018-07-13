import os
import pyltp

SENTENCE_DIR = "D:\\Projects\\Baike\\relation_sentences\\"
SENTENCE_ANNO_DIR = "D:\\Projects\\Baike\\sentences_annotation_auto\\"
SENTENCE_ANNO_DIR_SINGLE = SENTENCE_ANNO_DIR + "single_entity\\"
SENTENCE_ANNO_FILE_MULTIPLE_ENTITY = SENTENCE_ANNO_DIR + "annotation.txt"

LTP_DATA_DIR = "D:\\Projects\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = pyltp.Segmentor()
segmentor.load(cws_model_path)
postagger = pyltp.Postagger()
postagger.load(pos_model_path)
recognizer = pyltp.NamedEntityRecognizer()
recognizer.load(ner_model_path)

f = open(SENTENCE_ANNO_FILE_MULTIPLE_ENTITY, 'w', encoding="utf8")
count = 0
dirs = os.listdir(SENTENCE_DIR)
for file in dirs:
    filename = SENTENCE_DIR + file
    basename = os.path.basename(filename)
    print("文件名：" + basename)
    with open(filename, 'r', encoding="utf8")as g:
        basename_write_flag = False
        result_map = {}
        # {
        #   sentence_a :
        #   sentence_b: ...
        # }
        for line in g:
            line = line.strip()
            if len(line) == 0:
                continue
            # split relation and sentence
            split = line.rsplit(' ', 1)
            if len(split) < 2:
                continue
            relation = split[-1]
            sentence = split[0]
            if sentence in result_map:
                if relation in result_map[sentence]:
                    continue
                else:
                    result_map[sentence].add(relation)
            else:
                result_map[sentence] = set()
                result_map[sentence].add(relation)


            words = segmentor.segment(sentence)
            postags = postagger.postag(words)
            netags = recognizer.recognize(words, postags)

            entity_list = []
            for i in range(len(netags)):
                # if str(netags[i]).endswith("Nh")  and words[i] != data_dict['name'] and words[i] in PERSON_ENTITY_SET and words[i] not in ret_dict[rel]:
                if str(netags[i]).endswith("Nh"):
                     entity_list.append(words[i])
            entity_list_len = len(entity_list)
            if entity_list_len >= 2:
                for i in range(entity_list_len):
                    for j in range(i + 1, entity_list_len):
                        if not basename_write_flag:
                            f.write("@@{}\n".format(basename))
                            basename_write_flag = True

                        f.write("{} {} {} {}\n".format(sentence, entity_list[i], entity_list[j], relation,basename))
                        count += 1
    # if count >= 200:
    #     break
f.close()

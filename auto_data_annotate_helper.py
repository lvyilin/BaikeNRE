import os

CWD = os.getcwd()

ANNOTATION_DIR = CWD + "\\sentences_annotation_auto\\"
ANNOTATION_TARGET = ANNOTATION_DIR + "annotation.txt"
ANNOTATION_RESULT = ANNOTATION_DIR + "annotation_fin.txt"


def split_sentence(line):
    spl = str(line).rsplit(" ", 3)
    return spl[0]


with open(ANNOTATION_TARGET, "r", encoding="utf8") as f:
    lines = f.readlines()
    lines.reverse()
with open(ANNOTATION_RESULT, "r", encoding="utf8") as g:
    lines_resume = g.readlines()
    if len(lines_resume) == 0:
        last_line = ""
    else:
        last_line = lines_resume[-1]
with open(ANNOTATION_RESULT, "a", 1, encoding="utf8") as g:
    lines_len = len(lines)
    i = 0
    count = 0
    resume_flag = True
    if last_line == "":
        resume_flag = False
    while i < lines_len:
        # if count>10:
        #     break
        if resume_flag:
            if last_line == lines[i]:
                resume_flag = False
                i += 1
            if resume_flag:
                i += 1
                continue

        if lines[i].startswith("@@"):
            print(lines[i])
            i += 1
            continue
        print(lines[i])

        # single sentence mode
        # inp = input("Input 1 result: ")
        # if inp == '1':
        #     g.write((lines[i]))
        # i += 1
        # continue
        # end

        # multiple sentence mode
        sentence = split_sentence(lines[i])
        inc = 1
        while i + inc < lines_len:
            next_sentence = split_sentence(lines[i + inc])
            if next_sentence == sentence:
                print(lines[i + inc])
                inc += 1
            else:
                while True:
                    inp = input("Input {} result: ".format(inc))
                    inp = inp.rstrip("\n")
                    if len(inp) != inc:
                        if inp == "none":
                            inp = '2' * inc
                        else:
                            print("Input error.")
                            continue
                    for j in range(len(inp)):
                        if inp[j] == '1':
                            g.write(lines[i + j])
                            count += 1
                    break

                i += inc
                break

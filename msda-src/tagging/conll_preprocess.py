import random
import sys

class file_conll(object):

    def __init__(self, file_name):

        self.file_name = file_name
        f = open(self.file_name, "r")
        self.lines = f.readlines()
        f.close()
        self.data_x = []
        self.data_y = []
        self.tupled_data = []
        current_x = []
        current_y = []

        for line in self.lines:
            line = line.strip()
            if len(line) == 0:
                continue

            parts = line.split("\t")

            if parts[0].isdigit():
                if parts[0] == "1":
                    if len(current_x) > 0:
                        self.data_x.append(current_x)
                        self.data_y.append(current_y)
                        tuple_data = []

                        for x, y in zip(current_x, current_y):
                            tuple_data.append(tuple([x,y]))

                        self.tupled_data.append(tuple_data)
                        current_x = []
                        current_y = []

                current_x.append(parts[1])
                current_y.append(parts[3])

    def write_word_format(self, output_file):

        f = open(output_file, "w")
        for sentence, tags in zip(self.data_x, self.data_y):
            for word, tag in zip(sentence, tags):
                f.write(word+"|"+tag+" ")
            f.write("\n")
        f.close()

if __name__=="__main__":

    file_conll = file_conll(sys.argv[1])
    file_conll.write_word_format("something.txt")


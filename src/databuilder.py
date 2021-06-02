import os

def data_rebuild(in_name, out_name):
    fin = open(in_name, "r")
    fout = open(out_name, "w")
    for line in fin:
        content_list = line.split('\t')
        fout.write(content_list[-1])

if __name__ == '__main__':
    in_names = ["ID", "ID_test", "ID_train", "ID_validation",]
    in_dir = "../ISEAR/"
    in_base_name = "ISEAR "
    out_dir = "../data/"
    out_base_name = ""
    for in_name in in_names:
        data_rebuild(in_dir + in_base_name + in_name, out_dir + out_base_name + in_name)




import argparse
import torch
parser = argparse.ArgumentParser()

parser.add_argument('-file_ori_0', default="./data/political_data/democratic_only.dev.en", type=str,
                    help="""File name""")
parser.add_argument('-file_ori_1', default="./data/political_data/republican_only.dev.en", type=str,
                    help="""File name""")
parser.add_argument('-file_merge',default="./data/political_data/dev.merge",type=str,
                    help="""File merge""")
parser.add_argument('-file_label',default="./data/political_data/dev.labels",type=str,
                    help="""File label""")
parser.add_argument('-if_shuffle',default=0,type=int,
                    help="""File label""")
opt = parser.parse_args()

def main():
    file_0 = open(opt.file_ori_0,'r')
    file_1 = open(opt.file_ori_1,'r')
    file_merge = open(opt.file_merge,'a+')
    file_label = open(opt.file_label,'a+')

    file_set_0 = file_0.readlines()
    file_set_1 = file_1.readlines()
    print(len(file_set_0))
    print(len(file_set_1))
    file_set_total = file_set_0+file_set_1
    print(len(file_set_total))
    file_label_total = [0]*len(file_set_0)+[1]*len(file_set_1)

    if opt.if_shuffle == 1:
        perm = torch.randperm(len(file_set_total))
        file_set_new = [file_set_total[idx] for idx in perm]
        file_label_new = [file_label_total[idx] for idx in perm]
        for i in range(len(file_label_new)): 
            token_set = file_set_new[i].strip().split()
            if i % 5000 == 0:
                print("now:{0} total:{1}".format(i,len(file_set_new)))
            if len(token_set) > 16 or len(token_set) == 0:
                print("Over length")
                continue
            else:
                new_sentence = " ".join(token_set)
                file_merge.writelines(new_sentence.strip()+'\n')
                file_label.write(str(file_label_new[i])+'\n')
    else:
        for i in range(len(file_set_total)):
            token_set = file_set_total[i].strip().split()
            if len(token_set) > 15 or len(token_set) == 0:
                print("Over length")
                continue
            else:
                new_sentence = " ".join(token_set)
                file_merge.writelines(new_sentence.strip()+'\n')
                file_label.write(str(file_label_total[i])+'\n')
if __name__ == '__main__':
    main()

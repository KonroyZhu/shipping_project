import jieba.posseg as pseg

# print([c  for c in pseg.cut("本月国际二手油轮各个船型的价格整体略微有下滑趋势")])

def convert():
    tup_list=[]
    sentenct_list=open("../../../data/sentences.txt").readlines()
    tag_list=open("../../../data/tag.txt").readlines()
    for sen,tag in zip(sentenct_list,tag_list):
        sen=[c for c in pseg.cut(sen)]
        tags=[c for c in str(tag).strip().replace("\n","").split(" ")]
        sen_list=[]
        if len(tags)>1:
            for c in sen:
                c=tuple(c)
                if c[0] in tags:
                    c+=tuple(("T"))
                else:
                    c += tuple("O")
                sen_list.append(c)
        tup_list.append(sen_list)
        sen_list=[]
    return  tup_list


if __name__ == '__main__':
    tup_list=convert()
    print(tup_list)
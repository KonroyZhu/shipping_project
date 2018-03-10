from idx_project.idx_spider.config import Config
import requests
import re
from bs4 import BeautifulSoup

config = Config()


def find(regex, string):
    p = re.compile(regex)
    return str(float('%.3f' % float(re.findall(p, string)[0])))


def to_format(string):
    return str(float('%.3f' % float(string)))


def form_1_2():
    home_data = requests.post(config.home, headers=config.headers).text
    # 总指数
    line1_3 = find("原系统运价指数为(.*?),", home_data)
    line1_4 = find("校验系统运价指数为(.*?)，", home_data)
    print("总指数- -" + line1_3 + "-" + line1_4 + "- - ")

    dev1_data = requests.get(config.dev1, headers=config.headers).text

    def san_huo_ji_zhuang_xiang(name, option):
        line2 = re.findall(re.compile("data:\[(.*?)\]"), dev1_data)
        weigh = re.findall(re.compile("gd_weight = \[(.*?)]"), dev1_data)
        line2_2 = to_format(weigh[0].split(",")[option])
        line2_3 = to_format(line2[1].split(",")[option])
        line2_4 = to_format(line2[2].split(",")[option])
        print(name + "-" + line2_2 + "-" + line2_3 + "-" + line2_4 + "- - ")

    dev2_1_data = requests.post(config.dev2_1, headers=config.headers).text

    def ji_zhuag_xiang(name, option):
        line3 = re.findall(re.compile("data:\[(.*?)\]"), dev2_1_data)
        weigh2 = re.findall(re.compile("src_weight = \[(.*?)]"), dev2_1_data)
        line3_2 = to_format(weigh2[0].split(",")[option])
        line3_3 = to_format(line3[1].split(",")[option])
        line3_4 = to_format(line3[2].split(",")[option])
        print(name + "-" + line3_2 + "-" + line3_3 + "-" + line3_4 + "- - ")
        return line3_2

    dev2_2_data = requests.post(config.dev2_2, headers=config.headers).text

    def san_huo(name, option):
        line3 = re.findall(re.compile("data:\[(.*?)\]"), dev2_2_data)
        weigh2 = re.findall(re.compile("src_weight = \[(.*?)]"), dev2_2_data)
        line3_2 = to_format(weigh2[0].split(",")[option])
        line3_3 = to_format(line3[1].split(",")[option])
        line3_4 = to_format(line3[2].split(",")[option])
        print(name + "-" + line3_2 + "-" + line3_3 + "-" + line3_4 + "- - ")
        return line3_2

    # 集装箱指数
    san_huo_ji_zhuang_xiang("集装箱指数", 1)
    # 内河内贸
    w_1_6 = ji_zhuag_xiang("内河内贸", 0)
    # 香港航线
    w_2_6 = ji_zhuag_xiang("香港航线", 1)
    # 内河外贸
    w_3_6 = ji_zhuag_xiang("内河外贸", 2)
    # 散货指数
    san_huo_ji_zhuang_xiang("散货指数", 0)
    # 矿石
    w_1_2 = san_huo("矿石", 0)
    # 煤炭
    w_2_2 = san_huo("煤炭", 1)
    # 钢材
    w_3_2 = san_huo("钢材", 2)
    # 沙石自卸
    w_4_2 = san_huo("沙石自卸", 3)
    # 粮食
    w_5_2 = san_huo("粮食", 4)
    print()
    print("矿石-" + w_1_2 + "-1-	内河内贸-" + w_1_6 + "- - ")
    print("煤炭-" + w_2_2 + "-1-	香港航线-" + w_2_6 + "- - ")
    print("钢材-" + w_3_2 + "-1-	内河外贸-" + w_3_6 + "- - ")
    print("沙石自卸-" + w_4_2 + "-1-	 -" + " " + "- - ")
    print("粮食-" + w_5_2 + "-1-	 -" + " " + "- - ")


def form_3():
    abnormal_1_data = requests.post(config.abnormal_1, headers=config.headers).text
    td = re.findall(re.compile("<td>(.*?)</td>"), abnormal_1_data)
    hang_xiang = []
    bao_jia_ren = []
    yuan_bao_jia = []
    for i in range(len(td)):
        if (i % 5 == 0):
            hang_xiang.append(td[i + 1])
            bao_jia_ren.append(td[i + 2])
            yuan_bao_jia.append(td[i + 3].replace('<em><font color="red">', '').replace('</font></em>', ''))

    # print(abnormal_1_data)
    def bao_jia():
        soup = BeautifulSoup(abnormal_1_data, "lxml")
        bao_jia_l = soup.find_all(id="other")
        bao_jia_l_children = []
        for b in bao_jia_l:
            bao_jia_l_children.append(" ".join([c.string for c in b]).replace("\n", "").replace(",,", " ").strip())

        average = []
        for b in bao_jia_l_children:
            b = str(b).split("    ")
            num = [float(n.strip()) for n in b]
            average.append(to_format(sum(num) / len(num)))
        return bao_jia_l_children, average

    bao_jia, average = bao_jia()

    for i in range(len(hang_xiang)):
        print(str(hang_xiang[i]) + "-" + str(bao_jia[i]) + "-" + str(average[i]) + "-" + str(
            yuan_bao_jia[i]) + "- - - -" + str(bao_jia_ren[i]))


def form_4_5():
    data = requests.get(config.abnormal_2, headers=config.headers).text
    page = re.findall(re.compile('本期报价条数: <span style="color: green;">(.*?)</span>'), data)[0]

    url = "http://www.idxcheck.shippingex.cn/a/abnor/Reportlog?pageNo=1&pageSize=" + page
    abn_data = requests.get(url, headers=config.headers).text
    # print(abn_data)
    qi_shu = re.findall(re.compile('当前周期：<span style="color: green;">(.*?)</span>'), data)[0]
    bao_jia_shu = re.findall(re.compile('本期报价条数: <span style="color: green;">(.*?)</span>'), data)[0]
    yi_chang_shu = 0
    wei_bao_shu = 0

    soup = BeautifulSoup(abn_data, "lxml")
    lines = soup.find_all("tr")
    dan_wei_set = {}

    for l in lines:
        line = str(l).replace("\n", "").split("</td><td>")
        # print(len(line))
        if len(line) > 8:
            dan_wei = line[1]
            bao_jia_ren = line[2]
            bao_jia = line[6].replace('<font color="green"> ', '').replace(' </font>', '').replace(
                '<font color="red"> ', '')
            # print(dan_wei + " " + bao_jia_ren + " " + bao_jia)
            if dan_wei in dan_wei_set.keys():
                # bao_jia_ren-ying_bao-wei_bao-yi_chang
                content = str(dan_wei_set[dan_wei])
                ying_bao = int(content.split("-")[1])
                wei_bao = int(content.split("-")[2])
                yi_chang = int(content.split("-")[3])
                # print(bao_jia + "正常数据")
                if bao_jia == "正常数据":
                    ying_bao += 1
                elif bao_jia == "未上报数据":
                    wei_bao += 1
                    wei_bao_shu += 1
                elif bao_jia == "异常数据":
                    yi_chang += 1
                    yi_chang_shu += 1
                dan_wei_set[dan_wei] = bao_jia_ren + "-" + str(ying_bao) + "-" + str(wei_bao) + "-" + str(yi_chang)
            else:
                # bao_jia_ren-ying_bao-wei_bao-yi_chang
                ying_bao = 0
                wei_bao = 0
                yi_chang = 0
                if bao_jia == "正常数据":
                    ying_bao += 1
                elif bao_jia == "未上报数据":
                    wei_bao += 1
                    wei_bao_shu += 1
                elif bao_jia == "异常数据":
                    yi_chang += 1
                    yi_chang_shu += 1

                dan_wei_set[dan_wei] = bao_jia_ren + "-" + str(ying_bao) + "-" + str(wei_bao) + "-" + str(yi_chang)
    for key in dan_wei_set.keys():
        print(key + "-" + dan_wei_set[key])
    print()
    print(qi_shu + "-" + bao_jia_shu + "-" + str(to_format(str(float(yi_chang_shu) / float(bao_jia_shu)))) + "-" + str(
        wei_bao_shu)+"-" + str(to_format((float(bao_jia_shu)-(float(yi_chang_shu)+float(wei_bao_shu)))/float(bao_jia_shu))))


def form_7():
    print("test")


form_1_2()
print()
form_3()
print()
form_4_5()

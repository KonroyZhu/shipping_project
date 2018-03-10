class Config:
    def __init__(self):
        self.login = {"user": "fy", "pass": "123456"}
        self.headers = {"Host": "www.idxcheck.shippingex.cn",
                        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:35.0) Gecko/20100101 Firefox/35.0",
                        "Referer": "http://www.idxcheck.shippingex.cn/f/index;JSESSIONID=5786d520de184ce787116b9a775"
                                   "20fc8?dialog=true",
                        "Cookie": "Hm_lvt_82116c626a8d504a5c0675073362ef6f=1499659697,1499753513,1499843204,1500628141; JSESSIONID=CAE3EBC0FE811109AA3450E79FA01BA4; jeesite.session.id=6782537723a44f91a27c7237aeb08c8f; plisp_sso=97faec3e1dcd4442826907c84b06f503"}  # cookie have to be changed from time to time

        self.home = "http://www.idxcheck.shippingex.cn/a/gdufs/validate/idxcomposite/pageHomeIdxCompositeStatisticsLis" \
                    "t?idxId=00"
        self.dev1="http://www.idxcheck.shippingex.cn/a/gdufs/validate/sub/checksub?parentId=00&period=2018-7"
        self.dev2_1="http://www.idxcheck.shippingex.cn/a/gdufs/validate/linetype/checklinetype?idxId=00&&period=135&parentId=e595000ebbea4c85af6e13fe9caab5aa"
        self.dev2_2="http://www.idxcheck.shippingex.cn/a/gdufs/validate/linetype/checklinetype?idxId=00&&period=135&parentId=da9b4abd677d4467ab12ec3cbf1f9fdd"
        self.abnormal_1="http://www.idxcheck.shippingex.cn/ab/aa"
        self.abnormal_2="http://www.idxcheck.shippingex.cn/a/abnor/Reportlog"
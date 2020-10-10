import pandas as pd
import math


class CheckVariables:

    # GRSVT-4617
    # /Users/bokong/Documents/doc/20200927-GRSVT-4617/validate/FundingPartnerPaymentsAllInOneTrack.xlsx
    # /Users/bokong/Documents/doc/20200927-GRSVT-4617/validate/PartnerWithdrawalAllInOneTrack.xlsx
    # /Users/bokong/Documents/doc/20200927-GRSVT-4617/validate/FundingPartnerPaymentsAllInOneTrack_Component.xlsx

    file_path = '/Users/bokong/Documents/doc/20200927-GRSVT-4617/validate/FundingPartnerPaymentsAllInOneTrack_Component.xlsx'

    def __init__(self):
        pass

    def check(self):
        res = {'null':[],'one':[],'normal':[]}
        # read default first sheet
        df = pd.read_excel(self.file_path)
        # first 5 line
        # data = df.head()
        # print("获取到所有的值:\n{0}".format(data))
        for cl_name in df.columns:
            cl_values = pd.unique(df[cl_name]).tolist()
            if len(cl_values) == 1 and isinstance(cl_values[0], float) and math.isnan(cl_values[0]):
                res['null'].append(cl_name)
            elif len(cl_values) == 1:
                res['one'].append(cl_name)
            else:
                res['normal'].append(cl_name)
        pass
        print("null:%s, one:%s, normal:%s, total:%s\n" % (len(res['null'])-1, len(res['one']), len(res['normal'])-3,
              len(res['null'])-1+len(res['one'])+len(res['normal'])-3))
        print("null:\n")
        for s in res['null']:
            if s != "activity_id":
                print("%s" % s)
        print("one:\n")
        for s in res['one']:
            print("%s" % s)


def main():
    CheckVariables().check()


if __name__ == "__main__":
    main()

class Arithmet:
    def __init__(self):
        self.cumul_p = []

    def normalize_data(self, data):
        sum = 0
        for i in range(0, len(data)):
            sum += data[i]

        for i in range(0, len(data)):
            data[i] = data[i]/sum

        return data
    def cal_cumul_p(self, p):
        sum = 0
        for i in range(0, len(p)):
            sum += p[i]
            sum = round(sum, 3)
            self.cumul_p.append(sum)

    def arith(self, interval):
        len_interval = interval[1] - interval[0]
        # print(interval, len_interval)
        interval = [interval[0], len_interval*self.cumul_p[0]]
        len_interval = interval[1] - interval[0]
        # print(interval, len_interval)
        for i in range(1, len(self.cumul_p)):
            interval = [len_interval*self.cumul_p[i-1]+interval[0], len_interval*self.cumul_p[i]+interval[0]]
            len_interval = round(interval[1] - interval[0], 7)
            # print(interval, len_interval)

        arith_code = round((interval[0]+interval[1])/2, 7)
        return arith_code

if __name__ == '__main__':
    p = [0.1, 0.5, 0.1, 0.2, 0.1]
    test = Arithmet()
    test.cal_cumul_p(p)
    test.arith([0, 1])
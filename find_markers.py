import operator
def find_markers(data, Y, weight, len0, len1, len_dic, marker_numbers = 8 ,marker_length = 15):
    weight = weight[0:-2]
    new_weight = [sum(weight[i * len_dic: (i + 1) * len_dic]) for i in range(len(weight) / len_dic)]
    weight_seq1 = new_weight[0: len0]
    weight_seq2 = new_weight[len0:]
    
    result0 = []
    result1 = []
    for pos in range(3):
        mid_result0 = []
        mid_result1 = []
        for i in range(pos, len(weight_seq1), marker_length):
            temp = sum(weight_seq1[i: i + marker_length])
            mid_result0.append(Node(temp, i, i + marker_length - 1))
        for i in range(pos, len(weight_seq2), marker_length):
            temp = sum(weight_seq2[i: i + marker_length])
            mid_result1.append(Node(temp, i, i + marker_length - 1))
        mid_result0.sort(reverse = True, key = operator.attrgetter('sum'))
        mid_result1.sort(reverse = True, key = operator.attrgetter('sum'))
        result0.append(mid_result0[0: marker_numbers])
        result1.append(mid_result1[0: marker_numbers])
        
        statistics = {}
        for i in range(len(data)):
            s = data[i][0][mid_result0[0].begin: mid_result0[0].end + 1]
            if s not in statistics:
                statistics[s] = [0, Y[i]]
            statistics[s][0] += 1
        for key, value in statistics.iteritems():
            if value[0] > 1:
                print (key, value)
        print "wo shi fen ge xian"



#        print mid_result0[0: 9]
#    print result

class Node:
    sum = 0
    begin = -1
    end = -1
    
    def __init__(self, sum, begin, end):
        self.sum = sum
        self.begin = begin
        self.end = end

#    def __lt__(self, other):
#        return self.sum < other.sum
#
#    def __iter__(self):
#        return self



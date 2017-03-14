import operator


def find_markers(data, Y, weight, len0, len1, len_dic, marker_numbers=24, marker_length=15):
    weight = weight[0:-2]
    new_weight = [sum(weight[i * len_dic: (i + 1) * len_dic]) for i in range(len(weight) / len_dic)]
    weight_seq1 = new_weight[0: len0]
    weight_seq2 = new_weight[len0:]

    sorted_weight_seq1 = [(idx, val) for idx, val in enumerate(weight_seq1)]
    sorted_weight_seq1.sort(reverse=True, key=operator.itemgetter(1))
    sorted_weight_seq2 = [(idx, val) for idx, val in enumerate(weight_seq2)]
    sorted_weight_seq2.sort(reverse=True, key=operator.itemgetter(1))

    print sorted_weight_seq1[:20]
    print "fenge"
    print sorted_weight_seq2[:20]
    print "fenge"

    result0 = []
    result1 = []
    for i in range(len(weight_seq1) - marker_length + 1):
        temp = sum(weight_seq1[i: i + marker_length])
        result0.append(Node(temp, i, i + marker_length - 1))
    for i in range(len(weight_seq2) - marker_length + 1):
        temp = sum(weight_seq2[i: i + marker_length])
        result1.append(Node(temp, i, i + marker_length - 1))
    result0.sort(reverse=True, key=operator.attrgetter('sum'))
    result1.sort(reverse=True, key=operator.attrgetter('sum'))

    print [(r.sum, r.begin, r.end) for r in result0[:marker_numbers]]
    # print [(r.sum, r.begin, r.end) for r in result1[:marker_numbers]]
    print "fenge"

    posSubseq = []
    posSubseq1 = []
    negSubseq = []
    negSubseq1 = []

    for j in range(marker_numbers):
        statistics = {}
        statistics1 = {}
        for i in range(len(data)):
            s = data[i][0][result0[j].begin: result0[j].end + 1]
            s1 = data[i][1][result1[j].begin: result1[j].end + 1]
            if s not in statistics:
                statistics[s] = [0, 0, [], []]
            if Y[i] == 'better':
                statistics[s][0] += 1
                statistics[s][2].append(i)
            else:
                statistics[s][1] += 1
                statistics[s][3].append(i)

            if s1 not in statistics1:
                statistics1[s1] = [0, 0]
            if Y[i] == 'better':
                statistics1[s1][0] += 1
            else:
                statistics1[s1][1] += 1

        for key, value in statistics.iteritems():
            if value[0] > 3 and value[0] * 4 > value[1] * 1.5:
                posSubseq.append([key, result0[j].begin, result0[j].end])
            if value[1] > 12 and value[0] * 4 * 1.5 < value[1]:
                negSubseq.append([key, result0[j].begin, result0[j].end])

        for key, value in statistics1.iteritems():
            if value[0] > 3 and value[0] * 4 > value[1] * 1.5:
                posSubseq1.append([key, result1[j].begin, result1[j].end])
            if value[1] > 4.5 and value[0] * 4 * 1.5 < value[1]:
                negSubseq1.append([key, result1[j].begin, result1[j].end])
    # print len(posSubseq1)
    # posIdx = {}
    # posIdx1 = {}
    # for i in range(len(data)):
    #     if Y[i] == 'better':
    #         posIdx[i] = []
    #         posIdx1[i] = []
    #         for subseq in posSubseq:
    #             if data[i][0][subseq[1]:subseq[2]+1] == subseq[0]:
    #                 posIdx[i].append(subseq)
    #         for subseq in posSubseq1:
    #             if data[i][1][subseq[1]:subseq[2] + 1] == subseq[0]:
    #                 posIdx1[i].append(subseq)
    # for k, v in posIdx.iteritems():
    #     print (k, v)
    #     print posIdx1[k]
    return [posSubseq, posSubseq1, negSubseq, negSubseq1]



# print mid_result0[0: 9]
#    print result

class Node:
    sum = 0
    begin = -1
    end = -1

    def __init__(self, sum, begin, end):
        self.sum = sum
        self.begin = begin
        self.end = end

    def __str__(self):
        return "sum: %f, begin: %d, end %d" % (self.sum, self.begin, self.end)

# def __lt__(self, other):
#        return self.sum < other.sum
#
#    def __iter__(self):
#        return self



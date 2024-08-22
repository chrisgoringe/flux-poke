BAD = [
    1018720,
1135114,
1158742,
1199123,
1233515,
1265963,
1298962,
1310151,
1332861,
1340568,
1355537,
1393368,
1699386,
1731737,
1808640,
2054713,
2111704,
2436806,
2637835,
2699408,
2789006,
2806252,
3008511,
3286407,
3337307,
3364960,
3413469,
3443117,
3448117,
3659893,
3664764,
3674352,
3721812,
3899173,
4059699,
4102300,
4125701,
4139995,
4166979,
4218165,
4235931,
4421116,
4464822,
4553444,
4554065,
4597040,
4649668,
4654406,
4999673,
5061557,
5139757,
5161298,
5270527,
5298499,
5383785,
5475709,
5486208,
5556656,
5676993,
5783702,
5873525,
6040602,
6080422,
6116182,
6172036,
6214934,
6339076,
6437492,
6441834,
6512756,
6642743,
6646304,
6852597,
6908003,
6956457,
6979255,
7029558,
7078819,
7105710,
7219754,
7373207,
7452979,
7507585,
7797250,
7910159,
7961146,
8013900,
8260976,
8261631,
8543452,
8578703,
8587848,
8819356,
8975260,
9101529,
9136274,
9203861,
9287865,
9293340,
9567785,
9711542,
9864127,
9906778,
]

from modules.hffs import HFFS

h = HFFS('ChrisGoringe/fi')
for b in BAD:
    h.fs.rm(h.rpath(str(b)), recursive=True, maxdepth=1)
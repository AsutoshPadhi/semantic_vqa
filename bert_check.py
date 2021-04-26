from bert_score import score
import math

cands = []
refs = []

cands.append('India is my country')
cands.append('This is another sentence')
cands.append('India is my country')
cands.append('India is my country')
cands.append('India is my country')

refs.append('India is my country')
refs.append('This is another sentence')
refs.append('Near the tree')
refs.append('Hello world')
refs.append('France is my country')

P, R, F1 = score(cands, refs, lang="en", verbose=False)
# P = Variable(P, requires_grad=True)
# loss = -1 * torch.mean(P) * 10
# print(P, R, F1)
# print(type(P))
sum = 0
for s in P:
    sum += math.exp(s)

for p in P:
    print(p, (p*1000), 1-p)

# print()

# for p in P:
#     print((math.exp(p*20))/1000000)

# print()

# for p in P:
#     print((math.exp(p*10))/10)

# print()

# base = (math.exp(P[4]*10))/10

# for p in P:
#     print(((math.exp(p*10))/10)/base)

# print(math.exp(P[0])/sum)
# print(math.exp(P[1])/sum)
# print(math.exp(P[2])/sum)
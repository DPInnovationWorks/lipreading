import torch

letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def index2txt(indexs, start=1):
    pre = -1
    txt = []
    for n in indexs:
        if(pre != n and n >= start):                
            if(len(txt) > 0 and txt[-1] == ' ' and letters[n - start] == ' '):
                pass
            else:
                txt.append(letters[n - start])                
        pre = n
    return ''.join(txt).strip()
    

def ctc_decode(y):
    y = y.argmax(-1)
    return [index2txt(y[_], start=1) for _ in range(y.size(0))]


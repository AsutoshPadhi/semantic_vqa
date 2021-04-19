import torch

class ReversePaddedSequence():
    
    def rps(inputs, lengths, batch_first=False):
        if batch_first:
            inputs = inputs.transpose(0, 1)
        
        if inputs.size(1) != len(lengths):
            raise ValueError('inputs incompatible with lengths.')
        
        reversed_inputs = (inputs.data.clone())
        
        for i, length in enumerate(lengths):
            time_ind = torch.LongTensor(list(reversed(range(length))))
            reversed_inputs[:length, i] = inputs[:, i][time_ind]
        if batch_first:
            reversed_inputs = reversed_inputs.transpose(0, 1)
        
        return reversed_inputs
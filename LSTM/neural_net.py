import torch
from torch import nn

"""
This file for the LSTM design -- LSTM1:
- 2 layers of LSTM network
- 64 features in the hidden layer
"""


# Build a TimeDistributed class which allows to apply a layer to every temporal slice of an input.
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        # reshape input data into (samples * timesteps, input_size), i.e. squash timesteps
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))

        output = self.module(reshaped_input)
        # reshape the output, i.e. separate the timesteps
        if self.batch_first:
            # into (samples, timesteps, output_size)
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            # into (timesteps, samples, output_size)
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


# Build a LSTM network
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_directions = 1
        self.batch_first = True
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm_encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first)
        self.activation_fun1 = nn.ReLU()

        self.lstm_decoder = nn.LSTM(self.hidden_size, 64, self.num_layers, batch_first=self.batch_first)
        self.activation_fun2 = nn.ReLU()

        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.activation_fun3 = nn.LeakyReLU(negative_slope=0.3)

        self.timeDistributed = TimeDistributed(nn.Linear(self.output_size, self.output_size),
                                               batch_first=self.batch_first)

        self.device = device

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(1, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(1, batch_size, self.hidden_size).to(self.device)
        h_1 = torch.randn(1, batch_size, self.hidden_size).to(self.device)
        c_1 = torch.randn(1, batch_size, self.hidden_size).to(self.device)
        # output(batch_size, seq_len, num_directions * hidden_size) -- input(14, 20, 20)

        # only retain the last item of the encoded_output sequence (the last timestep)
        encoded_output, _ = self.lstm_encoder(input_seq, (h_0, c_0))  # output(14, 20, 64)
        encoded_output = self.activation_fun1(encoded_output)
        encoded_output = encoded_output[:, -1, :]  # output(14, 64)

        # repeat the last item 'seq_len' times (i.e. corresponds to RepeatVector module)
        repeated_output = encoded_output.unsqueeze(1).expand(-1, seq_len, -1)  # output(14, 20, 64)

        # get the decoded_output by using the lstm_decoder
        decoded_output, _ = self.lstm_decoder(repeated_output, (h_1, c_1))
        decoded_output = self.activation_fun2(decoded_output)  # output(14, 20, 64)

        # the fully connected layer
        prediction_output = self.linear(decoded_output)  # (14, 200)
        prediction_output = self.activation_fun3(prediction_output)

        # the TimeDistributed layer
        prediction_output = self.timeDistributed(prediction_output)

        return prediction_output

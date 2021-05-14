import torch


class ExHuneLSTM(torch.nn.Module):
    """Wrapper for multi-layer sequence forwarding via
       PhasedLSTMCell"""

    def __init__(
        self,
        input_size,
        hidden_size,
        alpha,
        bidirectional=True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.bi = 2 if bidirectional else 1

    def forward(self, u_sequence):
        """
        Args:
            sequence: The input sequence data of shape (batch, time, N)
            times: The timestamps corresponding to the data of shape (batch, time)
        """

        c0 = u_sequence.new_zeros((self.bi, u_sequence.size(0), self.hidden_size))
        h0 = u_sequence.new_zeros((self.bi, u_sequence.size(0), self.hidden_size))


        outputs = []
        for i in range(u_sequence.size(1)):
            u_t = u_sequence[:, i, :].unsqueeze(1)#u_sequence[:, i, :-1].unsqueeze(1)
            t_t = u_sequence[:, i, -1]

            lstm_out, (c_t, h_t) = self.lstm(u_t, (c0, h0))
            x_hat = lstm_out[:,:,-1].unsqueeze(1)
            x_hat_prime = u_t + x_hat
            lstm_next, (c_s, h_s) = self.lstm(x_hat_prime, (c_t, h_t))
            x_hat_next = lstm_next[:,:,-1].unsqueeze(1)
            out = u_t +(1-self.alpha)*x_hat_prime + x_hat_next * self.alpha
            c0, h0 = c_s, h_s

            outputs.append(out)
        outputs = torch.cat(outputs, dim=1)

        return outputs
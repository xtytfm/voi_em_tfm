import torch
import torch.nn as nn


class ModelCnnLstm(nn.Module):
    def __init__(self, num_classes,
                 in_linear=320,
                 hidden_size=128,
                 dropout=0.3,
                 dropout_lstm=0.1):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential()
        # 1. convolutinal block1
        self.add_conv_block(in_channels=1,
                            out_channels=16,
                            k_size_conv=3,
                            stride_conv=1,
                            padding=1,
                            k_size_pool=2,
                            stride_pool=2,
                            dropout=dropout)
        # 2. convolutinal block2
        self.add_conv_block(in_channels=16,
                            out_channels=32,
                            k_size_conv=3,
                            stride_conv=1,
                            padding=1,
                            k_size_pool=4,
                            stride_pool=4,
                            dropout=dropout)

        # 3. convolutinal block3
        self.add_conv_block(in_channels=32,
                            out_channels=64,
                            k_size_conv=3,
                            stride_conv=1,
                            padding=1,
                            k_size_pool=4,
                            stride_pool=4,
                            dropout=dropout)
        # 4. convolutinal block4
        self.add_conv_block(in_channels=64,
                            out_channels=64,
                            k_size_conv=3,
                            stride_conv=1,
                            padding=1,
                            k_size_pool=4,
                            stride_pool=4,
                            dropout=dropout)

        self.lstm_maxpool = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_lstm)
        self.attention_linear = nn.Linear(2 * hidden_size, 1)

        self.out_linear = nn.Linear(in_linear, num_classes)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.out_softmax = nn.Softmax(dim=1)

    def add_conv_block(self,
                       in_channels,
                       out_channels,
                       k_size_conv,
                       stride_conv,
                       padding,
                       k_size_pool,
                       stride_pool,
                       dropout):
        self.conv2Dblock.add_module(nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=k_size_conv,
                                              stride=stride_conv,
                                              padding=padding), )
        self.conv2Dblock.add_module(nn.BatchNorm2d(out_channels))
        self.conv2Dblock.add_module(nn.ReLU())
        self.conv2Dblock.add_module(nn.MaxPool2d(kernel_size=k_size_pool, stride=stride_pool))
        self.conv2Dblock.add_module(nn.Dropout(p=dropout))

    def forward(self, x):
        # conv features
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)
        # lstm features
        x_reduced = self.lstm_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        lstm_embedding, (h, c) = self.lstm(x_reduced)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        batch_size, T, _ = lstm_embedding.shape
        attention_weights = [None] * T
        for t in range(T):
            embedding = lstm_embedding[:, t, :]
            attention_weights[t] = self.attention_linear(embedding)
        attention_weights_norm = nn.functional.softmax(torch.stack(attention_weights, -1), -1)
        attention = torch.bmm(attention_weights_norm, lstm_embedding)
        attention = torch.squeeze(attention, 1)
        # concatenate features
        complete_embedding = torch.cat([conv_embedding, attention], dim=1)
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax, attention_weights_norm

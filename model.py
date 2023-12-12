import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod



class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self):
        """
        Forward pass logic
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)



class MLP(BaseModel):
    ''' Initialize the model '''
    def __init__(self, input_size, n_hidden_list, output_size=1, dropout_p=0, batch_norm=False):
        '''
        n_hidden_list: 각 hidden layer의 노드 수를 담은 리스트
        dropout_p: dropout 확률 (0~1, 0이면 dropout을 사용하지 않음)
        batch_norm: batch normalization 사용 여부
        '''
        super(MLP, self).__init__()

        self.fc_layers = nn.ModuleList(
            [nn.Linear(input_size, n_hidden_list[0])]
            + [nn.Linear(n_hidden_list[i], n_hidden_list[i+1]) for i in range(len(n_hidden_list)-1)]
            + [nn.Linear(n_hidden_list[-1], output_size)]
        )
        #L2 norm
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(n_hidden_list[i]) for i in range(len(n_hidden_list))]
            )
            
        self.dropout_p = dropout_p
        if dropout_p:
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(dropout_p) for _ in range(len(n_hidden_list))]
            )

        self.init_weights()

    ''' 순방향 전파 (함수 이름은 forward로 고정)'''
    def forward(self, x):
        # input x: (batch_size, input_size)
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i < len(self.fc_layers) - 1:
                if self.batch_norm:
                    x = self.bn_layers[i](x)
                x = F.relu(x)
                if self.dropout_p:
                    x = self.dropout_layers[i](x)

        return x  # (batch_size, output_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


import torch.nn as nn
import torch.nn.functional as F



if __name__ == '__main__':
    import torch
    
    batch_size = 7
    input_days = 5
    input_dim = 3

    model = MLP(input_size=input_days*input_dim,
                n_hidden_list=[512, 512, 512], 
                dropout_p=0.2, 
                batch_norm=True)
    print(model)

    x = torch.randn(batch_size, input_days, input_dim)
    y = model(x.reshape(x.size(0), -1))  # (batch_size, input_days * input_size) -> (batch_size, output_size)
    print(f'\nx: {x.shape} => y: {y.shape}')
import neurstitcher.nn as nn

from neurstitcher import rand
from neurstitcher.optimizer import MiniBatchSGD


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.LinearLayer(10, 3, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layer(x))

m = MyModule()
optim = MiniBatchSGD(m.parameters(), lr=0.01)

x = rand((5, 10))
for i in range(10):
    optim.zero_grad()
    y = m(x).sum()
    print(y)
    y.backward()
    optim.step()

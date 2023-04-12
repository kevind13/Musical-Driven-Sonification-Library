class Encoder(nn.Module):
    def __init__(self, code_size=1, input_shape=(128, 4, 89)):
        super(Encoder, self).__init__()        
        self.latent_dim = code_size
        self.input_shape = input_shape

        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, code_size)

        self.zero_mean = nn.BatchNorm1d(self.latent_dim, affine=False, eps=0)
        self.relu = nn.ReLU()
    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        z = x.view((batch_size, -1))

        z = self.zero_mean(z)
        return z  
    
class Decoder(nn.Module):
    def __init__(self, code_size=1, output_shape=(128, 4, 89)):
        super(Decoder, self).__init__()
        # Shape required to start transpose convs
        self.output_shape = output_shape

        output_size = output_shape[0] * output_shape[1] * output_shape[2]

        # Fully connected layers
        self.fc1 = nn.Linear(code_size, 512)
        self.fc2 = nn.Linear(512, output_size)

        self.relu = nn.ReLU()
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = self.relu(self.fc1(z))
        x = self.fc2(x)

        # Reshape output
        x = x.view(batch_size, *self.output_shape)
        x = nn.Sigmoid()(x)
        return x

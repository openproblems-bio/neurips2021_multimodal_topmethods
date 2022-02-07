""" autoencoder based models """
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """base encoder module"""

    def __init__(self, input_dim, out_dim, hidden_dim, dropout=0.2):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x_input):
        """forward propogation of the encoder arch"""
        x_emb = self.encoder(x_input)
        return x_emb


class Decoder(nn.Module):
    """base decoder module"""

    def __init__(self, input_dim, out_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x_emb):
        """forward propogation of the decoder arch"""
        x_rec = self.decoder(x_emb)
        return x_rec


class AutoEncoder(nn.Module):
    """autoencoder module"""

    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim, dropout=0.2):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hidden_dim, dropout)
        self.decoder = Decoder(feat_dim, out_dim, hidden_dim)

    def forward(self, x_input):
        """forward propogation of the autoencoder arch"""
        x_emb = self.encoder(x_input)
        x_rec = self.decoder(x_emb)
        return x_rec


class BatchClassifier(nn.Module):
    """base batch classifier class"""

    def __init__(self, input_dim, cls_num=6, hidden_dim=50):
        super(BatchClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, cls_num),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x_feat):
        """forward propogation of the batch classifier arch"""
        return self.classifier(x_feat)


class BatchRemovalGAN(nn.Module):
    """batch removal module"""

    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim, cls_num=10, dropout=0.2):
        super(BatchRemovalGAN, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hidden_dim, dropout)
        self.decoder = Decoder(feat_dim, out_dim, hidden_dim)
        self.classifier = BatchClassifier(feat_dim, cls_num=cls_num)

    def forward(self, x_input):
        """forward propogation of the batch removal gan arch"""
        x_feat = self.encoder(x_input)
        x_rec = self.decoder(x_feat)
        cls_prob = self.classifier(x_feat)

        return x_rec, cls_prob


if __name__ == "__main__":

    bsz = 5
    in_d = 10
    out_d = 3
    feat_d = 2
    hid_d = 10

    x1 = torch.randn(bsz, in_d).cuda()

    model = AutoEncoder(in_d, out_d, feat_d, hid_d).cuda().float()
    print(model)
    output = model(x1)
    print(output.shape)

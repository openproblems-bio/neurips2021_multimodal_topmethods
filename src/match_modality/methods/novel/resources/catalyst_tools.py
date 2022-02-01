import torch
import torch.nn.functional as F
from catalyst import runners, metrics
from models import symmetric_npair_loss


import numpy as np
import torch.nn.functional as F
from tqdm.notebook import tqdm

from networkx.algorithms import bipartite
from scipy import sparse



class scRNARunner(runners.Runner):
    def handle_batch(self, batch):
        features_first = batch['features_first']
        features_second = batch['features_second']

        logits, embeddings_first, embeddings_second = self.model(features_first, features_second)
        targets = torch.arange(logits.shape[0]).to(logits.device)
        
        loss = symmetric_npair_loss(logits, targets)
        
        batch_temperature = self.model.logit_scale.exp().item()
        
        self.batch_metrics.update({"loss": loss})
        self.batch_metrics.update({"T": batch_temperature})

        self.batch = {
                        'features_first': features_first,
                        'features_second': features_second,
                        'embeddings_first': embeddings_first,
                        'embeddings_second': embeddings_second,
                        'scores': logits, 
                        'targets': targets,
                        'temperature': batch_temperature

        }
        self.input = { 'features_first': features_first,
                       'features_second': features_second, 
                     }
        self.output = {'scores': logits,
                       'embeddings_first': embeddings_first,
                       'embeddings_second': embeddings_second
                      }
        
class CustomMetric(metrics.ICallbackLoaderMetric):
    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """Init."""
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""
        self.embeddings_list_first = []
        self.embeddings_list_second = []
        
    def reset(self, num_batches: int, num_samples: int) -> None:
        self.embeddings_list_first = []
        self.embeddings_list_second = []
        torch.cuda.empty_cache()
        
    def update(self, *args, **kwargs) -> None:
        embeddings_first = kwargs['embeddings_first']
        embeddings_second = kwargs['embeddings_second']
        temperature = kwargs['temperature']
        self.embeddings_list_first.append(temperature*embeddings_first)
        self.embeddings_list_second.append(embeddings_second)
        
    def compute(self):
        raise NotImplementedError('This method is not supported')
        
        
    def compute_key_value(self):
        all_embeddings_first = torch.cat(self.embeddings_list_first).detach().cpu()
        all_embeddings_second = torch.cat(self.embeddings_list_second).detach().cpu()
        logits = all_embeddings_first@all_embeddings_second.T
        #labels = torch.arange(logits.shape[0]).to(logits.device)
        labels = torch.arange(logits.shape[0])
       
        del(all_embeddings_first)
        del(all_embeddings_second)

        forward_accuracy = (torch.argmax(logits, dim=1)==labels).float().mean().item()
        backward_accuracy = (torch.argmax(logits, dim=0)==labels).float().mean().item()
        del(logits)

        avg_accuracy = 0.5*(forward_accuracy+backward_accuracy)
        
        loader_metrics = {

            'forward_acc':forward_accuracy,
            'backward_acc':backward_accuracy,
            'avg_acc': avg_accuracy
        }
        return loader_metrics
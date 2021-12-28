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
        #import ipdb; ipdb.set_trace()
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
        all_embeddings_first = torch.cat(self.embeddings_list_first)
        all_embeddings_second = torch.cat(self.embeddings_list_second)
        #all_embeddings_first = all_embeddings_first.cpu().detach()
        #all_embeddings_second = all_embeddings_second.cpu().detach()
        #import ipdb; ipdb.set_trace()
        logits = all_embeddings_first@all_embeddings_second.T
        #logits = logits.cpu().detach()
      
        #probabilities_forward = F.softmax(logits, dim=1)
        #probabilities_backward = F.softmax(logits, dim=0)
        labels = torch.arange(logits.shape[0]).to(logits.device)
       
        del(all_embeddings_first)
        del(all_embeddings_second)

        #forward_lb_metric = torch.diag(probabilities_forward).mean().item()*1000
        #backward_lb_metric = torch.diag(probabilities_backward).mean().item()*1000

        

        forward_accuracy = (torch.argmax(logits, dim=1)==labels).float().mean().item()
        backward_accuracy = (torch.argmax(logits, dim=0)==labels).float().mean().item()
        del(logits)
        #avg_lb_metric = 0.5*(forward_lb_metric+backward_lb_metric)
        avg_accuracy = 0.5*(forward_accuracy+backward_accuracy)
        
        loader_metrics = {
            #'lb_metric_forward':forward_lb_metric,
            #'lb_metric_backward':backward_lb_metric,
            #'lb_metric_avg':avg_lb_metric,
            'forward_acc':forward_accuracy,
            'backward_acc':backward_accuracy,
            'avg_acc': avg_accuracy
        }
        return loader_metrics
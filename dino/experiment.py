import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.prune as prune
import logging
import tqdm


class Experiment:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f'\t\tusing {self.device}')
        print(f'\t\tusing {self.device}')

        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        self.model.to(self.device).float()
        
        self.criterion = nn.CrossEntropyLoss()

    def run(self, train_loader, val_loader, path):
        metrics = {
            'top1_before': np.zeros(self.lesion_iters),
            'top5_before': np.zeros(self.lesion_iters),
            'top1_after': np.zeros(self.lesion_iters),
            'top5_after': np.zeros(self.lesion_iters)
        }
        
        modules_to_prune = [(module, 'weight') for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]
        
        for i in range(self.lesion_iters):
            print(f'\t\tIteration {i + 1}')
            logging.info(f'\t\tIteration {i + 1}')
            
            for module, parameter_name in modules_to_prune:
                prune.random_unstructured(module, name=parameter_name, amount=0.2)
                
            top1, top5 = self.test(val_loader)
            metrics['top1_before'][i] = top1
            metrics['top5_before'][i] = top5
            print(f'\t\t\tTop1: {top1:.5f}, Top5: {top5:.5f}')
            logging.info(f'\t\t\tTop1: {top1:.5f}, Top5: {top5:.5f}')

            self.retrain(train_loader)

            torch.save(self.model.state_dict(), path + '_ckpt_retrained_' + str(i) + '.pt')

            top1, top5 = self.test(val_loader)
            metrics['top1_after'][i] = top1
            metrics['top5_after'][i] = top5
            print(f'\t\t\tTop1: {top1:.5f}, Top5: {top5:.5f}')
            logging.info(f'\t\t\tTop1: {top1:.5f}, Top5: {top5:.5f}')

        return metrics

    def train(self, train_loader):
        self.model.train()
        
        head_optimizer = torch.optim.SGD(
                list(filter(lambda p: p.requires_grad, self.head.parameters())) + list(filter(lambda p: p.requires_grad, self.model.parameters())),
            lr=0.001,
            momentum=0.9,
            weight_decay=self.args['weight_decay'],
        )

        print('\t\t\tTraining...')
        logging.info('\t\t\tTraining...')

        iterator = iter(train_loader)    
        for it in range(self.retrain_epochs):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            head_optimizer.zero_grad()

            loss, acc = self.compute_metrics(self.head, inputs, targets)
            
            print(f'\t\t\tIter {it + 1}: train_loss={loss:.4f}, train_bacc={acc:.4f}')
            logging.info(f'\t\t\tIter {it + 1}: train_loss={loss:.4f}, train_bacc={acc:.4f}')
            
            loss.backward()
            head_optimizer.step()
            
    def accuracy(self, output, target, topk=(1,5)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
            return res

    def compute_metrics(self, inputs, targets):
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        top1, top5 = self.accuracy(output, targets)
        return loss, top1, top5

    def test(self, loader):
        print('\t\t\tTesting...')
        logging.info('\t\t\tTesting...')
                  
        self.model.eval()

        test_top1 = 0.0
        test_top5 = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                _, top1, top5 = self.compute_metrics(
                        inputs=inputs,
                        targets=targets,
                )

                test_top1 += top1
                test_top5 += top5

        return test_top1.detach().cpu().item() / len(loader.dataset),\
               test_top5.detach().cpu().item() / len(loader.dataset)

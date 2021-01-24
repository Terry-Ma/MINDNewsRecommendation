import torch

class NNOptimizer:
    def __init__(self, config, model):
        # init config
        self.config = config
        # init optimizer
        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                params=filter(lambda param: param.requires_grad, model.parameters()),
                lr=config['train']['lr']
                )
        else:
            self.optimizer = torch.optim.SGD(
                params=filter(lambda param: param.requires_grad, model.parameters()),
                lr=config['train']['lr'],
                momentum=config['train'].get('momentum', 0)
                )
        # init lr scheduler
        if config['train']['lr_decay']:
            lr_lambda = lambda step: min(step / config['train']['warmup_steps'], \
                (config['train']['train_steps'] - step) / \
                    (config['train']['train_steps'] - config['train']['warmup_steps']))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lr_lambda
                )
        self.lr = self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()
        if self.config['train']['lr_decay']:
            self.lr_scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']
    
    def zero_grad(self):
        self.optimizer.zero_grad()
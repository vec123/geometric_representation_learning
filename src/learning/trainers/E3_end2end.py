import torch
import torch.nn as nn
import torch.optim as optim
from src.learning.losses.losses import combined_surface_loss, kl_divergence_loss

class TrainingStepper:
    
    def __init__(self, encoder, decoder, learning_rate=1e-5):
        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()
        self.optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    def train_step(self, graph, true_verts, padding_mask):
        self.optimizer.zero_grad()
        
        # Forward
        inv_mean, inv_logvar, R_f, t_f = self.encoder(graph)
        inv = self.reparameterize(inv_mean, inv_logvar)
        pos_canonical = self.decoder(inv)
        
        # Loss (Assumes custom loss functions are implemented in torch)
        loss = combined_surface_loss(pos_canonical, true_verts, padding_mask) + \
               0.000 * kl_divergence_loss(inv_mean, inv_logvar)
        
        loss.backward()
        self.optimizer.step()
        return loss.item(), pos_canonical

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

class TrainingOrchestrator:
    def __init__(self, stepper, logger, dataloader):
        self.stepper = stepper
        self.logger = logger
        self.dataloader = dataloader
        self.loader = dataloader
        
    def run(self, num_steps, log_every=100, save_every=200):
        for step in range(num_steps):

            # 1. Fetch data
            batch = next(iter(self.loader))
            
            # 2. Perform training step (Stepper logic)
            state, loss, aux = self.stepper.train_step(batch)
            
            # 3. Handle logging (Logger logic)
            if step % log_every == 0:
                print(f"Step {step} | Loss: {loss:.6f}")
                self.logger.log_metrics({"loss": loss}, step)
                
            if step % save_every == 0:
                self.logger.save_checkpoint(state, step)
                self.logger.visualize_results(aux, step)
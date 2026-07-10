import torch
import torch.optim as optim

from src.learning.losses.losses import combined_surface_loss, kl_divergence_loss
from src.learning.models.encoder_output import EncoderOutput


def _resolve_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainingStepper:
    """Runs a single optimization step: encode -> reparameterize -> decode -> loss.

    Encoder and decoder are injected, so swapping model variants (equivariant vs
    not, supernodes vs full, different decoders) needs no change to this class.
    """

    def __init__(self, encoder, decoder, learning_rate=1e-5, kl_weight=0.0, device=None, verbose=False):
        self.device = _resolve_device(device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.kl_weight = kl_weight
        self.verbose = verbose
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )

    def encode(self, graph, supergraph):
        """Adapt the current GroupEncoder output into the standard EncoderOutput.

        NOTE: once encoders return an EncoderOutput directly this collapses to
        ``return self.encoder(graph)`` and any encoder becomes drop-in. This is the
        single seam that couples the trainer to the encoder's current signature.
        """
       
        out = self.encoder(
                graph, supergraph
                )
        
        assert isinstance(out, EncoderOutput)
        return out    

    def _forward(self, graph, super_graph, true_verts, padding_mask):
        """Encode -> latent -> decode -> loss, returning ``(loss_tensor, pred)``.

        The single forward path shared by ``train_step`` and ``eval_step`` so validation
        runs the exact same computation as training and can never diverge from it. This
        method does NOT touch the optimizer or gradients; the caller decides whether to
        backprop (train) or wrap the call in ``no_grad`` (eval)."""
        graph = graph.to(self.device)
        super_graph = super_graph.to(self.device) if super_graph is not None else None
        true_verts = true_verts.to(self.device)
        padding_mask = padding_mask.to(self.device)

        enc = self.encode(graph, super_graph)
        enc.mu = enc.mu.squeeze(1)

        latent =  self.reparameterize(enc.mu, enc.logvar)
        pred = self.decoder(latent)

        # The encoder's batch dim comes from graph.batch, which can desync from the
        # target if a shape lost all its nodes to dropout. Catch it loudly here instead
        # of letting torch.cdist broadcast (B_enc=1 vs B_target=N) into a wrong loss.
        if pred.shape[0] != true_verts.shape[0]:
            raise ValueError(
                f"batch mismatch: encoder produced {pred.shape[0]} shapes but target has "
                f"{true_verts.shape[0]}. A shape likely dropped all nodes during graph build."
            )

        recon_loss = combined_surface_loss(pred, true_verts, padding_mask)
        kl = enc.kl()
       
        kl_loss =  self.kl_weight * kl
        loss = recon_loss + kl_loss
   
        # A non-finite loss (e.g. NaN gradients from cdist at coincident points) would
        # otherwise train silently to all-NaN weights and write NaN VTPs without error.
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss ({loss.item()}); aborting.")

        return  pred, loss, recon_loss, kl_loss

    def train_step(self, graph, super_graph, true_verts, padding_mask):
        self.optimizer.zero_grad()
        pred, loss, recon, kl = self._forward(graph, super_graph, true_verts, padding_mask)
        loss.backward()
        self.optimizer.step()
        return  pred, loss.item(), recon, kl 

    @torch.no_grad()
    def eval_step(self, graph, super_graph, true_verts, padding_mask):
        """Validation forward pass: same computation as ``train_step`` but with no
        gradient tracking and no optimizer update. Set ``encoder``/``decoder`` to eval
        mode around this (the orchestrator does) so dropout/norm layers behave."""
        pred, loss, recon, kl = self._forward(graph, super_graph, true_verts, padding_mask)
        return pred, loss.item(), recon, kl 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class TrainingOrchestrator:
    """Drives the training loop: fetch a batch, step, log/checkpoint at cadence.

    The dataloader yields ``(graph, true_verts, padding_mask)`` batches, which are
    forwarded to ``stepper.train_step(*batch)``.
    """

    def __init__(self, stepper, logger, dataloader, val_loader=None):
        self.stepper = stepper
        self.logger = logger
        self.dataloader = dataloader
        self.val_loader = val_loader

    def run(self, num_steps, log_every=100, save_every=20, val_every=10):
        data_iter = iter(self.dataloader)
        for step in range(num_steps):

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            pred, loss,  recon, kl  = self.stepper.train_step(*batch)

            if step % log_every == 0:
                #print(f"Step {step} | Loss: {loss:.6f}")
                metrics = {
                    "loss": loss,
                    "recon": recon,
                    "kl":kl
                }
                self.logger.log_metrics(metrics, step)

            if step % save_every == 0:
                self.logger.save_checkpoint(self.stepper, step)
                self.logger.visualize_batch(batch, pred, step)
            if self.val_loader is not None and step % val_every == 0:
                self.run_validation(step)
                
        # Final metrics plot so the run always leaves a train/val curve behind, even if
        # the last step didn't land on a validation cadence.
        self.logger.plot_metrics()

    def run_validation(self, step, num_val_batches=1):
        """Evaluate on the validation loader without touching the optimizer, log the
        mean val loss, and save the validation predictions as VTPs.

        ``num_val_batches`` bounds how many batches to pull — the loaders here are
        infinite generators (``OneBatchLoader``/``ResamplingGraphLoader``), so iterating
        to exhaustion would never return. One batch covers the current single-batch
        validation set; raise it if a batched val loader is wired in later."""
        self.stepper.encoder.eval()
        self.stepper.decoder.eval()
        try:
            val_iter = iter(self.val_loader)
            val_losses, recon_val_losses, kl_val_losses = [], [], []

            last_batch, last_pred = None, None
            for _ in range(num_val_batches):
                batch = next(val_iter)
                pred,loss,  recon, kl = self.stepper.eval_step(*batch)
                val_losses.append(loss)
                recon_val_losses.append(recon)
                kl_val_losses.append(kl)
                
                last_batch, last_pred = batch, pred

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_recon_loss = sum(recon_val_losses) / len(val_losses)
            avg_kl_loss = sum(kl_val_losses) / len(val_losses)

            metrics = {
                    "avg_val_loss": avg_val_loss,
                    "avg_recon_loss": avg_recon_loss,
                    "avg_kl_loss":avg_kl_loss
                }
            
            #print(f"Step {step} | Val Loss: {avg_val_loss:.6f}")
            self.logger.log_metrics({"val_loss": avg_val_loss}, step)
            # Save the validation reconstructions (and inputs/targets) to a separate
            # subdir so they don't collide with the train-step VTPs at the same step.
            self.logger.visualize_val_batch(last_batch, last_pred, step)
            # Refresh the plot each validation so a long run shows live train-vs-val curves.
            self.logger.plot_metrics()
        finally:
            self.stepper.encoder.train()
            self.stepper.decoder.train()
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

    def __init__(self, encoder, decoder, learning_rate=1e-5, kl_weight=0.0, device=None):
        self.device = _resolve_device(device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.kl_weight = kl_weight
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

    def train_step(self, graph, super_graph, true_verts, padding_mask):
        graph = graph.to(self.device)
        super_graph = super_graph.to(self.device) if super_graph is not None else None
        true_verts = true_verts.to(self.device)
        padding_mask = padding_mask.to(self.device)

        self.optimizer.zero_grad()

        enc = self.encode(graph, super_graph)
        print("enc.mu.shape: ", enc.mu.shape)
        enc.mu = enc.mu.squeeze(1)
        print("enc.mu.shape: ", enc.mu.shape)
        print(torch.cdist(enc.mu, enc.mu))     
        z = torch.randn(4, enc.mu.shape[1], device=self.device) * 3   # 4 deliberately different codes
        out = self.decoder(z)
        print(torch.cdist(out.reshape(4, -1), out.reshape(4, -1)))  # ~0 => decoder ignores the latent

        latent = enc.mu   #self.reparameterize(enc.mu, enc.logvar)
        pred = self.decoder(latent)

        # The encoder's batch dim comes from graph.batch, which can desync from the
        # target if a shape lost all its nodes to dropout. Catch it loudly here instead
        # of letting torch.cdist broadcast (B_enc=1 vs B_target=N) into a wrong loss.
        if pred.shape[0] != true_verts.shape[0]:
            raise ValueError(
                f"batch mismatch: encoder produced {pred.shape[0]} shapes but target has "
                f"{true_verts.shape[0]}. A shape likely dropped all nodes during graph build."
            )

        loss = combined_surface_loss(pred, true_verts, padding_mask)
        kl = enc.kl()
        if kl is not None and self.kl_weight:
            loss = loss + self.kl_weight * kl

        # A non-finite loss (e.g. NaN gradients from cdist at coincident points) would
        # otherwise train silently to all-NaN weights and write NaN VTPs without error.
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss ({loss.item()}) at train_step; aborting.")

        loss.backward()
        self.optimizer.step()
        return loss.item(), pred

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class TrainingOrchestrator:
    """Drives the training loop: fetch a batch, step, log/checkpoint at cadence.

    The dataloader yields ``(graph, true_verts, padding_mask)`` batches, which are
    forwarded to ``stepper.train_step(*batch)``.
    """

    def __init__(self, stepper, logger, dataloader):
        self.stepper = stepper
        self.logger = logger
        self.dataloader = dataloader

    def run(self, num_steps, log_every=100, save_every=200):
        data_iter = iter(self.dataloader)
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            loss, pred = self.stepper.train_step(*batch)

            if step % log_every == 0:
                print(f"Step {step} | Loss: {loss:.6f}")
                self.logger.log_metrics({"loss": loss}, step)
            if step % save_every == 0:
                self.logger.save_checkpoint(self.stepper, step)
                self.logger.visualize_batch(batch, pred, step)

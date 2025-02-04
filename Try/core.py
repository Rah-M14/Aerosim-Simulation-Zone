import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pathlib import Path
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm

from loss import TrajLoss
from plotting import TrajectoryVisualizer

class NavigationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.control_net = nn.Sequential(
            nn.Linear(6, 16),  
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
            nn.Tanh()
        )
        
        # Initialize weights using Xavier initialization
        for layer in self.control_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, current_state: torch.Tensor) -> torch.Tensor:
        return self.control_net(current_state)

class NavigationTrainer:
    def __init__(
        self,
        world_bounds: Tuple[float, float] = (8.0, 6.0),
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        device: Optional[str] = None,
        project_name: str = "navigation_training",
        experiment_name: str = "Nav_pretrain"
    ):
        # Initialize wandb
        self.config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "world_bounds": world_bounds,
            "architecture": "NavigationNet",
            "optimizer": "AdamW"
        }
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=self.config
        )
        
        # Set device and parameters
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.world_bounds = torch.tensor(world_bounds, device=self.device)
        self.batch_size = batch_size
        
        # Initialize network and optimizer
        self.net = NavigationNet().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Initialize loss calculator and visualizer
        self.loss_calculator = TrajLoss(network=self.net_wrapper)
        self.visualizer = TrajectoryVisualizer(world_bounds=world_bounds)
        
        # Learning rate scheduler with cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-6
        )

    @torch.no_grad()
    def net_wrapper(self, numpy_input: np.ndarray) -> np.ndarray:
        """Vectorized wrapper for network predictions"""
        tensor_input = torch.from_numpy(numpy_input).float().to(self.device)
        if len(tensor_input.shape) == 1:
            tensor_input = tensor_input.unsqueeze(0)
        output = self.net(tensor_input)
        return output.cpu().numpy()

    def generate_batch(
        self, 
        batch_size: Optional[int] = None,
        min_distance: float = 0.5,
        max_distance: float = 12.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized batch generation"""
        batch_size = batch_size or self.batch_size
        
        # Generate all random values in one go
        initial_pos = torch.rand(batch_size, 2, device=self.device) * 2 - 1
        initial_pos *= self.world_bounds.unsqueeze(0)
        
        angles = torch.rand(batch_size, device=self.device) * (2 * np.pi) - np.pi
        distances = torch.rand(batch_size, device=self.device) * (max_distance - min_distance) + min_distance
        
        # Vectorized goal position calculation
        offsets = torch.stack([
            distances * torch.cos(angles),
            distances * torch.sin(angles)
        ], dim=1)
        
        goal_pos = initial_pos + offsets
        
        # Vectorized clamping
        goal_pos = torch.clamp(
            goal_pos,
            -self.world_bounds + 0.1,
            self.world_bounds - 0.1
        )
        
        return initial_pos.cpu().numpy(), goal_pos.cpu().numpy()

    def train_step(self, initial_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple[bool, float, Dict]:
        """Vectorized training step with metrics"""
        self.optimizer.zero_grad()
        
        # Vectorized loss calculation
        losses = []
        metrics = {
            "gradient_norm": 0.0,
            "loss_std": 0.0,
            "successful_gradients": 0
        }
        
        # Calculate losses in parallel using vectorized operations
        losses = [self.loss_calculator.simulate_trajectory(init, goal) 
                 for init, goal in zip(initial_pos, goal_pos)]
        losses = np.array(losses)
        
        # Calculate metrics
        avg_loss = np.mean(losses)
        metrics["loss_std"] = np.std(losses)
        metrics["successful_gradients"] = np.sum(np.isfinite(losses))
        
        if not np.isfinite(avg_loss):
            return False, float('inf'), metrics
        
        # Backpropagation
        loss_tensor = torch.tensor(avg_loss, requires_grad=True, device=self.device)
        loss_tensor.backward()
        
        # Calculate gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        metrics["gradient_norm"] = grad_norm.item()
        
        if any(p.grad is not None and not torch.all(torch.isfinite(p.grad)) 
               for p in self.net.parameters()):
            return False, float(avg_loss), metrics
        
        self.optimizer.step()
        return True, float(avg_loss), metrics

    def train(
        self, 
        epochs: int = 10000,
        log_interval: int = 500,
        save_interval: int = 1000,
        checkpoint_dir: str = "checkpoints",
        visualize_every: int = 1
    ):
        """Enhanced training loop with comprehensive logging and visualization"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        best_loss = float('inf')
        
        # Initialize progress bar
        pbar = tqdm(
            total=epochs,
            desc="Training Progress",
            dynamic_ncols=True,
            position=0,
            leave=True
        )
        
        # Initialize metrics for progress bar
        running_loss = 0.0
        running_success_rate = 0.0
        window_size = 100  # For moving average
        
        try:
            for epoch in range(epochs):
                # Generate and train
                initial_pos, goal_pos = self.generate_batch()
                success, loss, metrics = self.train_step(initial_pos, goal_pos)
                
                if not success:
                    wandb.log({"training_error": 1.0}, step=epoch)
                    continue
                
                # Update learning rate
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                
                # Update running metrics
                running_loss = 0.9 * running_loss + 0.1 * loss
                running_success_rate = (0.9 * running_success_rate + 
                                     0.1 * (metrics["successful_gradients"] / self.batch_size))
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'successful_gradients': f'{running_success_rate:.2%}',
                    'lr': f'{current_lr:.2e}'
                })
                pbar.update(1)
                
                # Log metrics
                log_dict = {
                    "loss": loss,
                    "learning_rate": current_lr,
                    "gradient_norm": metrics["gradient_norm"],
                    "loss_std": metrics["loss_std"],
                    "successful_gradients": metrics["successful_gradients"],
                    "success_rate": running_success_rate,
                    "epoch": epoch
                }
                
                # Visualization after each epoch
                if epoch % visualize_every == 0:
                    # Single trajectory visualization
                    test_initial, test_goal = self.generate_batch(batch_size=1)
                    single_traj_fig = self.visualizer.plot_trajectory(
                        self.net,
                        torch.from_numpy(test_initial[0]).float().to(self.device),
                        torch.from_numpy(test_goal[0]).float().to(self.device)
                    )
                    log_dict["single_trajectory"] = wandb.Image(single_traj_fig)
                    plt.close(single_traj_fig)
                    
                    # Multiple trajectories visualization
                    test_initial_multi, test_goal_multi = self.generate_batch(batch_size=5)
                    multi_traj_fig = self.visualizer.plot_multiple_trajectories(
                        self.net,
                        torch.from_numpy(test_initial_multi).float().to(self.device),
                        torch.from_numpy(test_goal_multi).float().to(self.device)
                    )
                    log_dict["multiple_trajectories"] = wandb.Image(multi_traj_fig)
                    plt.close(multi_traj_fig)
                    
                    # Save figures locally
                    single_traj_fig.savefig(checkpoint_dir / f'trajectory_epoch_{epoch}.png')
                    multi_traj_fig.savefig(checkpoint_dir / f'multiple_trajectories_epoch_{epoch}.png')
                
                # Log all metrics and visualizations
                wandb.log(log_dict, step=epoch)
                
                # Console output (in addition to progress bar)
                if epoch % log_interval == 0:
                    tqdm.write(f"Epoch {epoch}, Loss: {loss:.4f}, LR: {current_lr:.6f}")
                
                # Model saving
                if loss < best_loss:
                    best_loss = loss
                    self.save_model(checkpoint_dir / "best_model.pt")
                    wandb.run.summary["best_loss"] = best_loss
                
                if epoch % save_interval == 0:
                    self.save_model(checkpoint_dir / f"checkpoint_{epoch}.pt")
                    
        except KeyboardInterrupt:
            tqdm.write("\nTraining interrupted by user. Saving final model...")
            self.save_model(checkpoint_dir / "interrupted_model.pt")
            
        finally:
            pbar.close()
            
    def save_model(self, path: str):
        """Save model with metadata"""
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, path)
        # try:
            # Try to save to wandb, but don't fail if it doesn't work
            # wandb.save(str(path), base_path=str(Path(path).parent))
        # except Exception as e:
            # print(f"Warning: Could not save model to W&B: {e}")
            # Continue training even if wandb save fails
            # pass
    
    def load_model(self, path: str):
        """Load model with metadata"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

if __name__ == "__main__":
    # Initialize trainer
    trainer = NavigationTrainer(
        world_bounds=(8.0, 6.0),
        learning_rate=1e-3,
        batch_size=512
    )

    # Train the model
    trainer.train(
        epochs=10000,
        log_interval=500,
        save_interval=1000,
        checkpoint_dir="checkpoints",
        visualize_every=1
    )
